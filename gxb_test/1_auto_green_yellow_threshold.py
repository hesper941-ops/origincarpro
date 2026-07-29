#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
黄绿 HSV 阈值独立标定工具。

本工具只用于相机颜色阈值标定，不包含运动控制功能。它只订阅 ROS Image，
提供浏览器调参、自动采样以及 JSON/YAML 配置保存，不会访问相机设备文件。

USB 相机使用流程
================

终端 1 启动 USB 相机发布器：

    cd /root/intelligent_car_ws/src
    source /opt/tros/humble/setup.bash
    source /root/intelligent_car_ws/install/setup.bash

    python3 tools/usb_camera_publisher.py \
      --ros-args \
      -p device:=/dev/video0 \
      -p image_topic:=/image_out \
      -p width:=640 \
      -p height:=480 \
      -p fps:=15.0

终端 2 启动本工具：

    python3 gxb_test/1_auto_green_yellow_threshold.py \
      --ros-args \
      -p image_topic:=/image_out \
      -p web_gui_enable:=true \
      -p web_gui_port:=8089 \
      -p profile_name:=usb_camera

浏览器访问 http://小车IP:8089。请勿在本工具中再次打开 /dev/video0，
否则会与相机发布器争用设备。

深度相机使用流程
================

让深度相机 RGB 图像发布为 ROS Image，把 image_topic 改成现场实际 RGB 话题，
并使用独立 profile：

    python3 gxb_test/1_auto_green_yellow_threshold.py \
      --ros-args \
      -p image_topic:=/camera/color/image_raw \
      -p profile_name:=depth_camera \
      -p web_gui_enable:=true \
      -p web_gui_port:=8089

现场必须验证黄色、绿色初始阈值、ROI、形态学参数和底部连通偏好。
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


# 自动采样的黄色、绿色种子 Hue 范围刻意留出 35~41 的未知带，避免旧方案中
# 两种宽松种子大面积重叠而污染统计。
BROAD_YELLOW_LOWER = np.array((8, 45, 45), dtype=np.uint8)
BROAD_YELLOW_UPPER = np.array((34, 255, 255), dtype=np.uint8)
BROAD_GREEN_LOWER = np.array((42, 35, 30), dtype=np.uint8)
BROAD_GREEN_UPPER = np.array((100, 255, 255), dtype=np.uint8)

CONFIG_VERSION = 1
PROFILE_ALLOWED_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
)


DEFAULT_CONFIG: Dict[str, object] = {
    # 以下 HSV 值仅为现场调参起点，必须根据相机和光照验证。
    "yellow_h_min": 15,
    "yellow_h_max": 40,
    "yellow_s_min": 70,
    "yellow_s_max": 255,
    "yellow_v_min": 70,
    "yellow_v_max": 255,
    "green_h_min": 35,
    "green_h_max": 90,
    "green_s_min": 55,
    "green_s_max": 255,
    "green_v_min": 45,
    "green_v_max": 255,
    "roi_y_start_ratio": 0.35,
    "roi_y_end_ratio": 1.0,
    "open_kernel_size": 3,
    "close_kernel_size": 7,
    "open_iterations": 1,
    "close_iterations": 2,
    "min_component_area": 300,
    "bottom_band_pixels": 24,
    "yellow_prefer_bottom_connected": True,
    "green_prefer_bottom_connected": False,
    "auto_sample_frames": 120,
    "auto_max_samples_per_color_per_frame": 5000,
    "auto_low_percentile": 2.0,
    "auto_high_percentile": 98.0,
    "auto_hue_margin": 3,
    "auto_saturation_margin": 12,
    "auto_value_margin": 12,
    "auto_min_total_samples_per_color": 1500,
}


PARAMETER_SCHEMA: Dict[str, Tuple[str, object, object, object, str]] = {
    "yellow_h_min": ("黄色 HSV", 0, 179, 1, "number"),
    "yellow_h_max": ("黄色 HSV", 0, 179, 1, "number"),
    "yellow_s_min": ("黄色 HSV", 0, 255, 1, "number"),
    "yellow_s_max": ("黄色 HSV", 0, 255, 1, "number"),
    "yellow_v_min": ("黄色 HSV", 0, 255, 1, "number"),
    "yellow_v_max": ("黄色 HSV", 0, 255, 1, "number"),
    "green_h_min": ("绿色 HSV", 0, 179, 1, "number"),
    "green_h_max": ("绿色 HSV", 0, 179, 1, "number"),
    "green_s_min": ("绿色 HSV", 0, 255, 1, "number"),
    "green_s_max": ("绿色 HSV", 0, 255, 1, "number"),
    "green_v_min": ("绿色 HSV", 0, 255, 1, "number"),
    "green_v_max": ("绿色 HSV", 0, 255, 1, "number"),
    "roi_y_start_ratio": ("ROI", 0.0, 0.95, 0.01, "number"),
    "roi_y_end_ratio": ("ROI", 0.05, 1.0, 0.01, "number"),
    "open_kernel_size": ("形态学", 1, 15, 2, "number"),
    "close_kernel_size": ("形态学", 1, 21, 2, "number"),
    "open_iterations": ("形态学", 0, 5, 1, "number"),
    "close_iterations": ("形态学", 0, 5, 1, "number"),
    "min_component_area": ("形态学", 0, 50000, 10, "number"),
    "bottom_band_pixels": ("形态学", 1, 100, 1, "number"),
    "yellow_prefer_bottom_connected": (
        "形态学", False, True, None, "bool"
    ),
    "green_prefer_bottom_connected": (
        "形态学", False, True, None, "bool"
    ),
    "auto_sample_frames": ("自动采样", 1, 1000, 1, "number"),
    "auto_max_samples_per_color_per_frame": (
        "自动采样", 1, 50000, 100, "number"
    ),
    "auto_low_percentile": ("自动采样", 0.0, 99.0, 0.5, "number"),
    "auto_high_percentile": ("自动采样", 1.0, 100.0, 0.5, "number"),
    "auto_hue_margin": ("自动采样", 0, 30, 1, "number"),
    "auto_saturation_margin": ("自动采样", 0, 100, 1, "number"),
    "auto_value_margin": ("自动采样", 0, 100, 1, "number"),
    "auto_min_total_samples_per_color": (
        "自动采样", 1, 1000000, 100, "number"
    ),
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
class ThresholdConfig:
    """保存运行时可调参数，并负责完整组合校验。"""

    values: Dict[str, object] = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_CONFIG)
    )

    @staticmethod
    def _odd_kernel(value: object) -> int:
        """把核尺寸限制为正奇数。"""
        size = max(1, int(round(float(value))))
        return size if size % 2 else size + 1

    @classmethod
    def validate(cls, candidate: Dict[str, object]) -> Dict[str, object]:
        """验证完整候选配置，成功时返回规范化副本。"""
        cfg = copy.deepcopy(candidate)
        cfg["open_kernel_size"] = cls._odd_kernel(cfg["open_kernel_size"])
        cfg["close_kernel_size"] = cls._odd_kernel(
            cfg["close_kernel_size"]
        )

        def require(condition: bool, message: str, parameter: str) -> None:
            if not condition:
                raise APIError(message, parameter)

        for color in ("yellow", "green"):
            for channel, maximum in (("h", 179), ("s", 255), ("v", 255)):
                low_name = f"{color}_{channel}_min"
                high_name = f"{color}_{channel}_max"
                low = int(cfg[low_name])
                high = int(cfg[high_name])
                require(
                    0 <= low <= high <= maximum,
                    f"{low_name} 不能大于 {high_name}，且必须位于合法范围",
                    low_name,
                )
        roi_start = float(cfg["roi_y_start_ratio"])
        roi_end = float(cfg["roi_y_end_ratio"])
        require(
            0.0 <= roi_start < roi_end <= 1.0,
            "ROI 必须满足 0.0 <= start < end <= 1.0",
            "roi_y_start_ratio",
        )
        require(
            roi_end - roi_start >= 0.10 - 1e-9,
            "roi_y_end_ratio 必须至少比 roi_y_start_ratio 大 0.10",
            "roi_y_end_ratio",
        )
        require(
            int(cfg["open_kernel_size"]) >= 1
            and int(cfg["close_kernel_size"]) >= 1,
            "形态学核尺寸必须大于等于 1",
            "open_kernel_size",
        )
        require(
            int(cfg["open_kernel_size"]) <= 15
            and int(cfg["close_kernel_size"]) <= 21,
            "开运算核不能超过 15，闭运算核不能超过 21",
            "open_kernel_size",
        )
        require(
            int(cfg["open_iterations"]) >= 0
            and int(cfg["close_iterations"]) >= 0
            and int(cfg["open_iterations"]) <= 5
            and int(cfg["close_iterations"]) <= 5,
            "形态学迭代次数必须位于 0 到 5",
            "open_iterations",
        )
        require(
            0 <= int(cfg["min_component_area"]) <= 50000,
            "min_component_area 必须位于 0 到 50000",
            "min_component_area",
        )
        require(
            1 <= int(cfg["bottom_band_pixels"]) <= 100,
            "bottom_band_pixels 必须位于 1 到 100",
            "bottom_band_pixels",
        )
        low_percentile = float(cfg["auto_low_percentile"])
        high_percentile = float(cfg["auto_high_percentile"])
        require(
            0.0 <= low_percentile < high_percentile <= 100.0,
            "自动采样百分位必须满足 0 <= low < high <= 100",
            "auto_low_percentile",
        )
        require(
            int(cfg["auto_sample_frames"]) > 0,
            "auto_sample_frames 必须大于 0",
            "auto_sample_frames",
        )
        require(
            int(cfg["auto_max_samples_per_color_per_frame"]) > 0,
            "每帧最大采样数必须大于 0",
            "auto_max_samples_per_color_per_frame",
        )
        require(
            int(cfg["auto_min_total_samples_per_color"]) > 0,
            "最小总样本数必须大于 0",
            "auto_min_total_samples_per_color",
        )
        return cfg

    def snapshot(self) -> Dict[str, object]:
        """返回不会被其他线程修改的参数副本。"""
        return copy.deepcopy(self.values)


@dataclass
class RuntimeState:
    """Web 状态、最新图像和编码帧。"""

    image_received: bool = False
    image_topic: str = "/image_out"
    frame_width: int = 0
    frame_height: int = 0
    encoding: str = ""
    fps: float = 0.0
    last_frame_time: float = 0.0
    yellow_pixels: int = 0
    green_pixels: int = 0
    overlap_pixels: int = 0
    unknown_pixels: int = 0
    boundary_preview_pixels: int = 0
    profile_name: str = "usb_camera"
    last_saved_path: str = ""
    last_loaded_path: str = ""
    last_error: str = ""
    latest_bgr: Optional[np.ndarray] = None
    encoded_frames: Dict[str, bytes] = field(default_factory=dict)

    def status(self, cfg: Dict[str, object]) -> Dict[str, object]:
        """生成字段稳定的状态响应。"""
        age = (
            max(0.0, time.monotonic() - self.last_frame_time)
            if self.last_frame_time > 0.0
            else None
        )
        return {
            "image_received": self.image_received,
            "image_topic": self.image_topic,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "encoding": self.encoding,
            "fps": self.fps,
            "last_frame_age_sec": age,
            "roi_y_start_ratio": cfg["roi_y_start_ratio"],
            "roi_y_end_ratio": cfg["roi_y_end_ratio"],
            "yellow_pixels": self.yellow_pixels,
            "green_pixels": self.green_pixels,
            "overlap_pixels": self.overlap_pixels,
            "unknown_pixels": self.unknown_pixels,
            "boundary_preview_pixels": self.boundary_preview_pixels,
            "profile_name": self.profile_name,
            "last_saved_path": self.last_saved_path,
            "last_loaded_path": self.last_loaded_path,
            "last_error": self.last_error,
        }


@dataclass
class AutoCalibrationState:
    """保存自动采样进度、样本与建议阈值。"""

    sampling: bool = False
    sampled_frames: int = 0
    yellow_batches: List[np.ndarray] = field(default_factory=list)
    green_batches: List[np.ndarray] = field(default_factory=list)
    suggested_yellow: Optional[Dict[str, int]] = None
    suggested_green: Optional[Dict[str, int]] = None
    yellow_median_hsv: Optional[List[float]] = None
    green_median_hsv: Optional[List[float]] = None
    yellow_percentile_low_hsv: Optional[List[float]] = None
    yellow_percentile_high_hsv: Optional[List[float]] = None
    green_percentile_low_hsv: Optional[List[float]] = None
    green_percentile_high_hsv: Optional[List[float]] = None
    message: str = "尚未开始自动采样"

    def reset(self) -> None:
        """释放旧数组并恢复初始采样状态。"""
        self.sampling = False
        self.sampled_frames = 0
        self.yellow_batches.clear()
        self.green_batches.clear()
        self.suggested_yellow = None
        self.suggested_green = None
        self.yellow_median_hsv = None
        self.green_median_hsv = None
        self.yellow_percentile_low_hsv = None
        self.yellow_percentile_high_hsv = None
        self.green_percentile_low_hsv = None
        self.green_percentile_high_hsv = None
        self.message = "采样已清空"

    @staticmethod
    def sample_count(batches: List[np.ndarray]) -> int:
        """统计分批数组中的像素样本总数。"""
        return sum(int(batch.shape[0]) for batch in batches)

    def status(self, requested_frames: int) -> Dict[str, object]:
        """生成自动采样状态。"""
        return {
            "auto_sampling": self.sampling,
            "auto_sampled_frames": self.sampled_frames,
            "auto_requested_frames": requested_frames,
            "auto_green_sample_count": self.sample_count(
                self.green_batches
            ),
            "auto_yellow_sample_count": self.sample_count(
                self.yellow_batches
            ),
            "auto_result_ready": (
                self.suggested_yellow is not None
                and self.suggested_green is not None
            ),
            "green_median_hsv": self.green_median_hsv,
            "yellow_median_hsv": self.yellow_median_hsv,
            "green_percentile_low_hsv": self.green_percentile_low_hsv,
            "green_percentile_high_hsv": self.green_percentile_high_hsv,
            "yellow_percentile_low_hsv": self.yellow_percentile_low_hsv,
            "yellow_percentile_high_hsv": self.yellow_percentile_high_hsv,
            "suggested_green": self.suggested_green,
            "suggested_yellow": self.suggested_yellow,
            "auto_message": self.message,
        }


class ImageCodec:
    """不依赖 cv_bridge 的 ROS Image/BGR 转换器。"""

    CHANNELS = {
        "bgr8": 3,
        "rgb8": 3,
        "bgra8": 4,
        "rgba8": 4,
        "mono8": 1,
    }

    @classmethod
    def to_bgr(cls, msg: Image) -> np.ndarray:
        """处理行填充并把受支持编码统一转换为 BGR。"""
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
    def encode_jpeg(image: np.ndarray, quality: int) -> bytes:
        """按原分辨率编码 JPEG，不主动缩放。"""
        ok, encoded = cv2.imencode(
            ".jpg",
            np.ascontiguousarray(image),
            [cv2.IMWRITE_JPEG_QUALITY, int(np.clip(quality, 1, 100))],
        )
        if not ok:
            raise ValueError("JPEG 编码失败")
        return encoded.tobytes()


@dataclass
class ProcessResult:
    """单帧颜色分割结果。"""

    raw: np.ndarray
    overlay: np.ndarray
    yellow_mask: np.ndarray
    green_mask: np.ndarray
    unknown_mask: np.ndarray
    hsv: np.ndarray
    roi_start: int
    roi_end: int
    overlap_pixels: int
    boundary_preview_pixels: int


class ThresholdProcessor:
    """执行 ROI 内 HSV 分割、形态学和连通域筛选。"""

    @staticmethod
    def _morph(mask: np.ndarray, cfg: Dict[str, object]) -> np.ndarray:
        """依次进行开、闭运算。"""
        output = mask
        open_iterations = int(cfg["open_iterations"])
        close_iterations = int(cfg["close_iterations"])
        if open_iterations > 0:
            open_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (int(cfg["open_kernel_size"]),) * 2,
            )
            output = cv2.morphologyEx(
                output,
                cv2.MORPH_OPEN,
                open_kernel,
                iterations=open_iterations,
            )
        if close_iterations > 0:
            close_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (int(cfg["close_kernel_size"]),) * 2,
            )
            output = cv2.morphologyEx(
                output,
                cv2.MORPH_CLOSE,
                close_kernel,
                iterations=close_iterations,
            )
        return output

    @staticmethod
    def _filter_components(
        mask: np.ndarray,
        min_area: int,
        prefer_bottom: bool,
        bottom_band_pixels: int,
    ) -> np.ndarray:
        """按面积筛选连通域，并可优先保留触及 ROI 底部的区域。"""
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        retained = [
            label
            for label in range(1, count)
            if int(stats[label, cv2.CC_STAT_AREA]) >= min_area
        ]
        if prefer_bottom and retained:
            band_start = max(0, mask.shape[0] - bottom_band_pixels)
            bottom_labels = set(
                int(value)
                for value in np.unique(labels[band_start:, :])
                if int(value) > 0
            )
            connected = [
                label for label in retained if label in bottom_labels
            ]
            if connected:
                retained = connected
        output = np.zeros_like(mask)
        for label in retained:
            output[labels == label] = 255
        return output

    @classmethod
    def seed_mask(
        cls,
        hsv_roi: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        cfg: Dict[str, object],
        prefer_bottom: bool,
    ) -> np.ndarray:
        """为自动采样生成经过清理的宽松种子 mask。"""
        mask = cv2.inRange(hsv_roi, lower, upper)
        mask = cls._morph(mask, cfg)
        return cls._filter_components(
            mask,
            int(cfg["min_component_area"]),
            prefer_bottom,
            int(cfg["bottom_band_pixels"]),
        )

    @classmethod
    def process(
        cls, bgr: np.ndarray, cfg: Dict[str, object]
    ) -> ProcessResult:
        """处理完整图像，但只在 ROI 内产生颜色 mask。"""
        height, width = bgr.shape[:2]
        roi_start = int(height * float(cfg["roi_y_start_ratio"]))
        roi_end = int(height * float(cfg["roi_y_end_ratio"]))
        roi_start = min(max(roi_start, 0), height - 1)
        roi_end = min(max(roi_end, roi_start + 1), height)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hsv_roi = hsv[roi_start:roi_end, :]

        yellow_lower = np.array(
            (
                cfg["yellow_h_min"],
                cfg["yellow_s_min"],
                cfg["yellow_v_min"],
            ),
            dtype=np.uint8,
        )
        yellow_upper = np.array(
            (
                cfg["yellow_h_max"],
                cfg["yellow_s_max"],
                cfg["yellow_v_max"],
            ),
            dtype=np.uint8,
        )
        green_lower = np.array(
            (
                cfg["green_h_min"],
                cfg["green_s_min"],
                cfg["green_v_min"],
            ),
            dtype=np.uint8,
        )
        green_upper = np.array(
            (
                cfg["green_h_max"],
                cfg["green_s_max"],
                cfg["green_v_max"],
            ),
            dtype=np.uint8,
        )

        yellow_roi = cv2.inRange(hsv_roi, yellow_lower, yellow_upper)
        green_roi = cv2.inRange(hsv_roi, green_lower, green_upper)
        yellow_roi = cls._filter_components(
            cls._morph(yellow_roi, cfg),
            int(cfg["min_component_area"]),
            bool(cfg["yellow_prefer_bottom_connected"]),
            int(cfg["bottom_band_pixels"]),
        )
        green_roi = cls._filter_components(
            cls._morph(green_roi, cfg),
            int(cfg["min_component_area"]),
            bool(cfg["green_prefer_bottom_connected"]),
            int(cfg["bottom_band_pixels"]),
        )

        overlap = cv2.bitwise_and(yellow_roi, green_roi)
        not_overlap = cv2.bitwise_not(overlap)
        yellow_roi = cv2.bitwise_and(yellow_roi, not_overlap)
        green_roi = cv2.bitwise_and(green_roi, not_overlap)
        unknown_roi = cv2.bitwise_not(
            cv2.bitwise_or(yellow_roi, green_roi)
        )

        yellow = np.zeros((height, width), dtype=np.uint8)
        green = np.zeros((height, width), dtype=np.uint8)
        unknown = np.zeros((height, width), dtype=np.uint8)
        yellow[roi_start:roi_end, :] = yellow_roi
        green[roi_start:roi_end, :] = green_roi
        unknown[roi_start:roi_end, :] = unknown_roi

        raw_preview = bgr.copy()
        cv2.line(
            raw_preview,
            (0, roi_start),
            (width - 1, roi_start),
            (255, 255, 0),
            1,
        )
        cv2.line(
            raw_preview,
            (0, min(roi_end, height - 1)),
            (width - 1, min(roi_end, height - 1)),
            (255, 255, 0),
            1,
        )
        overlay = bgr.copy()
        # ROI 外轻微变暗但仍保留完整视野。
        overlay[:roi_start, :] = (
            overlay[:roi_start, :].astype(np.float32) * 0.55
        ).astype(np.uint8)
        overlay[roi_end:, :] = (
            overlay[roi_end:, :].astype(np.float32) * 0.55
        ).astype(np.uint8)
        tint = overlay.copy()
        tint[yellow > 0] = (0, 255, 255)
        tint[green > 0] = (0, 210, 0)
        cv2.addWeighted(tint, 0.42, overlay, 0.58, 0.0, overlay)
        cv2.line(overlay, (0, roi_start), (width - 1, roi_start), (255, 255, 0), 1)
        cv2.line(
            overlay,
            (0, min(roi_end, height - 1)),
            (width - 1, min(roi_end, height - 1)),
            (255, 255, 0),
            1,
        )

        # 接触预览只表示两种 mask 的邻近位置，不参与任何路径计算。
        contact_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5)
        )
        yellow_dilated = cv2.dilate(yellow, contact_kernel, iterations=1)
        green_dilated = cv2.dilate(green, contact_kernel, iterations=1)
        boundary = cv2.bitwise_and(yellow_dilated, green_dilated)
        overlay[boundary > 0] = (255, 0, 255)

        return ProcessResult(
            raw=raw_preview,
            overlay=overlay,
            yellow_mask=yellow,
            green_mask=green,
            unknown_mask=unknown,
            hsv=hsv,
            roi_start=roi_start,
            roi_end=roi_end,
            overlap_pixels=int(cv2.countNonZero(overlap)),
            boundary_preview_pixels=int(cv2.countNonZero(boundary)),
        )


class ConfigStore:
    """负责 profile 路径、JSON/YAML 保存和加载。"""

    def __init__(self, script_dir: Path) -> None:
        self.config_dir = script_dir / "config"
        self.config_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate_profile(profile_name: str) -> str:
        """拒绝绝对路径、点号和目录穿越字符。"""
        if (
            not profile_name
            or ".." in profile_name
            or any(
                character not in PROFILE_ALLOWED_CHARS
                for character in profile_name
            )
        ):
            raise APIError(
                "profile_name 只允许字母、数字、下划线和短横线",
                "profile_name",
            )
        return profile_name

    def profile_paths(self, profile_name: str) -> Tuple[Path, Path]:
        """返回安全 profile 对应的 JSON/YAML 文件。"""
        safe_name = self.validate_profile(profile_name)
        stem = f"{safe_name}_green_yellow"
        return self.config_dir / f"{stem}.json", self.config_dir / f"{stem}.yaml"

    @staticmethod
    def _yaml_scalar(value: object) -> str:
        """生成无需 PyYAML 的可读 YAML 标量。"""
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        return json.dumps(str(value), ensure_ascii=False)

    @classmethod
    def _yaml_lines(
        cls, value: object, indent: int = 0
    ) -> List[str]:
        """递归输出固定结构的 YAML。"""
        prefix = " " * indent
        if not isinstance(value, dict):
            return [prefix + cls._yaml_scalar(value)]
        lines: List[str] = []
        for key, item in value.items():
            if isinstance(item, dict):
                lines.append(f"{prefix}{key}:")
                lines.extend(cls._yaml_lines(item, indent + 2))
            else:
                lines.append(
                    f"{prefix}{key}: {cls._yaml_scalar(item)}"
                )
        return lines

    def save(
        self,
        payload: Dict[str, object],
        profile_name: str,
        explicit_path: str,
    ) -> Tuple[Path, Path]:
        """原子性较好的临时文件替换方式保存 JSON 和 YAML。"""
        if explicit_path:
            json_path = Path(explicit_path).expanduser().resolve()
            if json_path.suffix.lower() != ".json":
                raise APIError("config_path 必须是 JSON 文件", "config_path")
            yaml_path = json_path.with_suffix(".yaml")
            json_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            json_path, yaml_path = self.profile_paths(profile_name)
        json_text = json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=False
        ) + "\n"
        yaml_text = "\n".join(self._yaml_lines(payload)) + "\n"
        json_tmp = json_path.with_suffix(json_path.suffix + ".tmp")
        yaml_tmp = yaml_path.with_suffix(yaml_path.suffix + ".tmp")
        json_tmp.write_text(json_text, encoding="utf-8")
        yaml_tmp.write_text(yaml_text, encoding="utf-8")
        json_tmp.replace(json_path)
        yaml_tmp.replace(yaml_path)
        return json_path, yaml_path

    def load_path(self, profile_name: str, explicit_path: str) -> Path:
        """按显式路径优先规则选择加载文件。"""
        if explicit_path:
            return Path(explicit_path).expanduser().resolve()
        return self.profile_paths(profile_name)[0]


class GreenYellowThresholdNode(Node):
    """ROS 图像订阅与标定状态协调节点。"""

    def __init__(self) -> None:
        super().__init__("green_yellow_threshold_calibrator")
        node_defaults: Dict[str, object] = {
            "image_topic": "/image_out",
            "web_gui_enable": True,
            "web_gui_host": "0.0.0.0",
            "web_gui_port": 8089,
            "web_gui_jpeg_quality": 92,
            "web_gui_max_fps": 6.0,
            "config_path": "",
            "profile_name": "usb_camera",
            "auto_load_config": True,
        }
        all_defaults = {**node_defaults, **DEFAULT_CONFIG}
        for name, value in all_defaults.items():
            self.declare_parameter(name, value)
        ros_values = {
            name: self.get_parameter(name).value for name in all_defaults
        }

        self.cfg_lock = threading.RLock()
        self.state_lock = threading.RLock()
        self.auto_lock = threading.RLock()
        self.frame_condition = threading.Condition(self.state_lock)
        self.config = ThresholdConfig(
            ThresholdConfig.validate(
                {name: ros_values[name] for name in DEFAULT_CONFIG}
            )
        )
        self.image_topic = str(ros_values["image_topic"])
        self.web_gui_enable = bool(ros_values["web_gui_enable"])
        self.web_gui_host = str(ros_values["web_gui_host"])
        self.web_gui_port = int(ros_values["web_gui_port"])
        self.jpeg_quality = int(
            np.clip(ros_values["web_gui_jpeg_quality"], 1, 100)
        )
        self.web_max_fps = max(
            0.1, float(ros_values["web_gui_max_fps"])
        )
        self.explicit_config_path = str(ros_values["config_path"])
        self.auto_load_config = bool(ros_values["auto_load_config"])
        self.store = ConfigStore(Path(__file__).resolve().parent)
        try:
            self.profile_name = self.store.validate_profile(
                str(ros_values["profile_name"])
            )
        except APIError as exc:
            self.profile_name = "usb_camera"
            self.get_logger().error(str(exc))

        self.state = RuntimeState(
            image_topic=self.image_topic,
            profile_name=self.profile_name,
        )
        self.auto = AutoCalibrationState()
        self.last_callback_time = 0.0
        self.last_encode_time = 0.0
        self.http_server: Optional[ThreadingHTTPServer] = None
        self.http_thread: Optional[threading.Thread] = None

        if self.auto_load_config:
            try:
                self.load_config(require_exists=False)
            except Exception as exc:
                with self.state_lock:
                    self.state.last_error = f"自动加载配置失败：{exc}"
                self.get_logger().warning(self.state.last_error)

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(Image, self.image_topic, self._on_image, qos)
        if self.web_gui_enable:
            self._start_web_server()
        self.get_logger().info(
            f"黄绿阈值标定工具已启动：image_topic={self.image_topic}, "
            f"profile={self.profile_name}"
        )

    def cfg_snapshot(self) -> Dict[str, object]:
        """取得一帧内一致的配置快照。"""
        with self.cfg_lock:
            return self.config.snapshot()

    def update_parameter(self, name: str, raw_value: str) -> Dict[str, object]:
        """复制、完整校验后原子替换运行时参数。"""
        if name == "profile_name":
            profile = self.store.validate_profile(str(raw_value))
            with self.state_lock:
                self.profile_name = profile
                self.state.profile_name = profile
            return {"ok": True, "name": name, "value": profile}
        if name not in PARAMETER_SCHEMA:
            raise APIError(f"参数不可调：{name}", name)
        schema = PARAMETER_SCHEMA[name]
        with self.cfg_lock:
            current = self.config.snapshot()
            old_value = current[name]
            try:
                if schema[4] == "bool":
                    lowered = str(raw_value).strip().lower()
                    if lowered not in ("true", "false", "1", "0"):
                        raise ValueError
                    value: object = lowered in ("true", "1")
                elif isinstance(old_value, int) and not isinstance(
                    old_value, bool
                ):
                    value = int(round(float(raw_value)))
                else:
                    value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise APIError(f"{name} 的值无效", name) from exc
            minimum, maximum = schema[1], schema[2]
            if minimum is not None and (
                float(value) < float(minimum)
                or float(value) > float(maximum)
            ):
                raise APIError(
                    f"{name} 必须位于 {minimum} 到 {maximum}", name
                )
            current[name] = value
            validated = ThresholdConfig.validate(current)
            self.config = ThresholdConfig(validated)
        return {"ok": True, "name": name, "value": validated[name]}

    def restore_defaults(self) -> Dict[str, object]:
        """仅恢复运行时默认值，不写磁盘。"""
        with self.cfg_lock:
            self.config = ThresholdConfig(
                ThresholdConfig.validate(DEFAULT_CONFIG)
            )
        return {"ok": True, "message": "已恢复内置默认值，尚未保存"}

    def _on_image(self, msg: Image) -> None:
        """解码并处理最新相机帧，错误只更新状态。"""
        now = time.monotonic()
        try:
            bgr = ImageCodec.to_bgr(msg)
            cfg = self.cfg_snapshot()
            result = ThresholdProcessor.process(bgr, cfg)
            self._collect_auto_samples(result, cfg)
            elapsed = now - self.last_callback_time
            instantaneous_fps = 1.0 / elapsed if elapsed > 1e-6 else 0.0
            self.last_callback_time = now
            with self.state_lock:
                previous_fps = self.state.fps
                self.state.fps = (
                    instantaneous_fps
                    if previous_fps <= 0.0
                    else 0.15 * instantaneous_fps + 0.85 * previous_fps
                )
                self.state.image_received = True
                self.state.frame_width = int(bgr.shape[1])
                self.state.frame_height = int(bgr.shape[0])
                self.state.encoding = str(msg.encoding)
                self.state.last_frame_time = now
                self.state.latest_bgr = bgr.copy()
                self.state.yellow_pixels = int(
                    cv2.countNonZero(result.yellow_mask)
                )
                self.state.green_pixels = int(
                    cv2.countNonZero(result.green_mask)
                )
                self.state.overlap_pixels = result.overlap_pixels
                self.state.unknown_pixels = int(
                    cv2.countNonZero(result.unknown_mask)
                )
                self.state.boundary_preview_pixels = (
                    result.boundary_preview_pixels
                )
                self.state.last_error = ""
            if (
                self.web_gui_enable
                and now - self.last_encode_time >= 1.0 / self.web_max_fps
            ):
                self._encode_web_frames(result)
                self.last_encode_time = now
        except Exception as exc:
            with self.state_lock:
                self.state.last_error = f"图像处理失败：{exc}"
                self.state.last_frame_time = now

    def _encode_web_frames(self, result: ProcessResult) -> None:
        """编码五种原分辨率画面并一次性替换共享帧。"""
        frames = {
            "raw": ImageCodec.encode_jpeg(result.raw, self.jpeg_quality),
            "overlay": ImageCodec.encode_jpeg(
                result.overlay, self.jpeg_quality
            ),
            "yellow": ImageCodec.encode_jpeg(
                cv2.cvtColor(result.yellow_mask, cv2.COLOR_GRAY2BGR),
                self.jpeg_quality,
            ),
            "green": ImageCodec.encode_jpeg(
                cv2.cvtColor(result.green_mask, cv2.COLOR_GRAY2BGR),
                self.jpeg_quality,
            ),
            "unknown": ImageCodec.encode_jpeg(
                cv2.cvtColor(result.unknown_mask, cv2.COLOR_GRAY2BGR),
                self.jpeg_quality,
            ),
        }
        with self.frame_condition:
            self.state.encoded_frames = frames
            self.frame_condition.notify_all()

    @staticmethod
    def _limited_samples(
        hsv_roi: np.ndarray, mask: np.ndarray, maximum: int
    ) -> np.ndarray:
        """随机限制单帧样本数，避免采样内存无限增长。"""
        pixels = hsv_roi[mask > 0]
        if pixels.shape[0] <= maximum:
            return pixels.copy()
        indices = np.random.default_rng().choice(
            pixels.shape[0], size=maximum, replace=False
        )
        return pixels[indices].copy()

    def _collect_auto_samples(
        self, result: ProcessResult, cfg: Dict[str, object]
    ) -> None:
        """自动采样开启时，从不重叠种子范围累积 HSV。"""
        with self.auto_lock:
            if not self.auto.sampling:
                return
            hsv_roi = result.hsv[result.roi_start:result.roi_end, :]
            yellow_seed = ThresholdProcessor.seed_mask(
                hsv_roi,
                BROAD_YELLOW_LOWER,
                BROAD_YELLOW_UPPER,
                cfg,
                bool(cfg["yellow_prefer_bottom_connected"]),
            )
            green_seed = ThresholdProcessor.seed_mask(
                hsv_roi,
                BROAD_GREEN_LOWER,
                BROAD_GREEN_UPPER,
                cfg,
                bool(cfg["green_prefer_bottom_connected"]),
            )
            maximum = int(
                cfg["auto_max_samples_per_color_per_frame"]
            )
            yellow = self._limited_samples(hsv_roi, yellow_seed, maximum)
            green = self._limited_samples(hsv_roi, green_seed, maximum)
            if yellow.size:
                self.auto.yellow_batches.append(yellow)
            if green.size:
                self.auto.green_batches.append(green)
            self.auto.sampled_frames += 1
            self.auto.message = (
                f"正在采样 {self.auto.sampled_frames}/"
                f"{cfg['auto_sample_frames']}"
            )
            if self.auto.sampled_frames >= int(cfg["auto_sample_frames"]):
                self.auto.sampling = False
                self._calculate_suggestions_locked(cfg)

    @staticmethod
    def _suggest_from_samples(
        samples: np.ndarray,
        low_percentile: float,
        high_percentile: float,
        margins: Tuple[int, int, int],
    ) -> Tuple[Dict[str, int], List[float]]:
        """用分位数和 margin 生成合法 HSV 上下限。"""
        low = np.percentile(samples, low_percentile, axis=0)
        high = np.percentile(samples, high_percentile, axis=0)
        margin = np.asarray(margins, dtype=float)
        lower = np.floor(low - margin)
        upper = np.ceil(high + margin)
        lower = np.clip(lower, (0, 0, 0), (179, 255, 255)).astype(int)
        upper = np.clip(upper, (0, 0, 0), (179, 255, 255)).astype(int)
        suggestion = {
            "h_min": int(lower[0]),
            "h_max": int(upper[0]),
            "s_min": int(lower[1]),
            "s_max": int(upper[1]),
            "v_min": int(lower[2]),
            "v_max": int(upper[2]),
        }
        median = [
            round(float(value), 2)
            for value in np.median(samples, axis=0)
        ]
        return suggestion, median

    def _calculate_suggestions_locked(
        self, cfg: Dict[str, object]
    ) -> None:
        """样本足够时计算建议值；不足时保留手动参数。"""
        yellow_count = AutoCalibrationState.sample_count(
            self.auto.yellow_batches
        )
        green_count = AutoCalibrationState.sample_count(
            self.auto.green_batches
        )
        minimum = int(cfg["auto_min_total_samples_per_color"])
        missing = []
        if yellow_count < minimum:
            missing.append(f"黄色样本不足({yellow_count}/{minimum})")
        if green_count < minimum:
            missing.append(f"绿色样本不足({green_count}/{minimum})")
        if missing:
            self.auto.suggested_yellow = None
            self.auto.suggested_green = None
            self.auto.message = "；".join(missing)
            return
        yellow_samples = np.concatenate(self.auto.yellow_batches, axis=0)
        green_samples = np.concatenate(self.auto.green_batches, axis=0)
        low_percentile = float(cfg["auto_low_percentile"])
        high_percentile = float(cfg["auto_high_percentile"])
        self.auto.yellow_percentile_low_hsv = [
            round(float(value), 2)
            for value in np.percentile(
                yellow_samples, low_percentile, axis=0
            )
        ]
        self.auto.yellow_percentile_high_hsv = [
            round(float(value), 2)
            for value in np.percentile(
                yellow_samples, high_percentile, axis=0
            )
        ]
        self.auto.green_percentile_low_hsv = [
            round(float(value), 2)
            for value in np.percentile(
                green_samples, low_percentile, axis=0
            )
        ]
        self.auto.green_percentile_high_hsv = [
            round(float(value), 2)
            for value in np.percentile(
                green_samples, high_percentile, axis=0
            )
        ]
        margins = (
            int(cfg["auto_hue_margin"]),
            int(cfg["auto_saturation_margin"]),
            int(cfg["auto_value_margin"]),
        )
        self.auto.suggested_yellow, self.auto.yellow_median_hsv = (
            self._suggest_from_samples(
                yellow_samples,
                low_percentile,
                high_percentile,
                margins,
            )
        )
        self.auto.suggested_green, self.auto.green_median_hsv = (
            self._suggest_from_samples(
                green_samples,
                low_percentile,
                high_percentile,
                margins,
            )
        )
        self.auto.message = "建议阈值已生成，请检查后点击应用"

    def auto_start(self) -> Dict[str, object]:
        """清空旧样本并开始新的多帧采样。"""
        with self.auto_lock:
            self.auto.reset()
            self.auto.sampling = True
            self.auto.message = "等待采集图像"
        return {"ok": True, "message": "自动采样已开始"}

    def auto_stop(self) -> Dict[str, object]:
        """停止采样并尝试用现有样本计算建议。"""
        cfg = self.cfg_snapshot()
        with self.auto_lock:
            self.auto.sampling = False
            self._calculate_suggestions_locked(cfg)
            message = self.auto.message
        return {"ok": True, "message": message}

    def auto_reset(self) -> Dict[str, object]:
        """清空所有自动采样数组和建议。"""
        with self.auto_lock:
            self.auto.reset()
        return {"ok": True, "message": "采样和建议已清空"}

    def auto_apply(self) -> Dict[str, object]:
        """仅在用户明确点击时，把建议阈值写入运行时配置。"""
        with self.auto_lock:
            yellow = copy.deepcopy(self.auto.suggested_yellow)
            green = copy.deepcopy(self.auto.suggested_green)
        if yellow is None or green is None:
            raise APIError("尚无可应用的完整建议阈值")
        with self.cfg_lock:
            candidate = self.config.snapshot()
            for color, suggestion in (
                ("yellow", yellow),
                ("green", green),
            ):
                for key, value in suggestion.items():
                    candidate[f"{color}_{key}"] = value
            self.config = ThresholdConfig(
                ThresholdConfig.validate(candidate)
            )
        return {"ok": True, "message": "建议阈值已应用，尚未保存"}

    def auto_status(self) -> Dict[str, object]:
        """线程安全地取得自动采样状态。"""
        cfg = self.cfg_snapshot()
        with self.auto_lock:
            status = self.auto.status(int(cfg["auto_sample_frames"]))
        status.update({
            "auto_low_percentile": cfg["auto_low_percentile"],
            "auto_high_percentile": cfg["auto_high_percentile"],
            "auto_hue_margin": cfg["auto_hue_margin"],
            "auto_saturation_margin": cfg["auto_saturation_margin"],
            "auto_value_margin": cfg["auto_value_margin"],
        })
        return status

    def _config_payload(self) -> Dict[str, object]:
        """生成 JSON/YAML 共用的固定配置结构。"""
        cfg = self.cfg_snapshot()
        with self.state_lock:
            state = copy.deepcopy(self.state.status(cfg))
        with self.auto_lock:
            auto = copy.deepcopy(
                self.auto.status(int(cfg["auto_sample_frames"]))
            )
        return {
            "version": CONFIG_VERSION,
            "profile_name": self.profile_name,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "image_topic": self.image_topic,
            "frame_width": state["frame_width"],
            "frame_height": state["frame_height"],
            "roi": {
                "y_start_ratio": cfg["roi_y_start_ratio"],
                "y_end_ratio": cfg["roi_y_end_ratio"],
            },
            "yellow": {
                "h_min": cfg["yellow_h_min"],
                "h_max": cfg["yellow_h_max"],
                "s_min": cfg["yellow_s_min"],
                "s_max": cfg["yellow_s_max"],
                "v_min": cfg["yellow_v_min"],
                "v_max": cfg["yellow_v_max"],
                "prefer_bottom_connected": cfg[
                    "yellow_prefer_bottom_connected"
                ],
            },
            "green": {
                "h_min": cfg["green_h_min"],
                "h_max": cfg["green_h_max"],
                "s_min": cfg["green_s_min"],
                "s_max": cfg["green_s_max"],
                "v_min": cfg["green_v_min"],
                "v_max": cfg["green_v_max"],
                "prefer_bottom_connected": cfg[
                    "green_prefer_bottom_connected"
                ],
            },
            "morphology": {
                "open_kernel_size": cfg["open_kernel_size"],
                "close_kernel_size": cfg["close_kernel_size"],
                "open_iterations": cfg["open_iterations"],
                "close_iterations": cfg["close_iterations"],
                "min_component_area": cfg["min_component_area"],
                "bottom_band_pixels": cfg["bottom_band_pixels"],
            },
            "auto_calibration": {
                "sample_frames": cfg["auto_sample_frames"],
                "max_samples_per_color_per_frame": cfg[
                    "auto_max_samples_per_color_per_frame"
                ],
                "low_percentile": cfg["auto_low_percentile"],
                "high_percentile": cfg["auto_high_percentile"],
                "hue_margin": cfg["auto_hue_margin"],
                "saturation_margin": cfg["auto_saturation_margin"],
                "value_margin": cfg["auto_value_margin"],
                "min_total_samples_per_color": cfg[
                    "auto_min_total_samples_per_color"
                ],
                "yellow_sample_count": auto["auto_yellow_sample_count"],
                "green_sample_count": auto["auto_green_sample_count"],
                "suggested_yellow": auto["suggested_yellow"],
                "suggested_green": auto["suggested_green"],
            },
            "statistics": {
                "yellow_pixels": state["yellow_pixels"],
                "green_pixels": state["green_pixels"],
                "overlap_pixels": state["overlap_pixels"],
                "unknown_pixels": state["unknown_pixels"],
                "fps": state["fps"],
            },
        }

    @staticmethod
    def _flat_config_from_payload(
        payload: Dict[str, object]
    ) -> Dict[str, object]:
        """把保存的固定嵌套结构还原为运行时平面参数。"""
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        roi = payload.get("roi", {})
        yellow = payload.get("yellow", {})
        green = payload.get("green", {})
        morphology = payload.get("morphology", {})
        auto = payload.get("auto_calibration", {})
        if not all(
            isinstance(section, dict)
            for section in (roi, yellow, green, morphology, auto)
        ):
            raise APIError("配置文件结构错误")
        cfg["roi_y_start_ratio"] = roi.get(
            "y_start_ratio", cfg["roi_y_start_ratio"]
        )
        cfg["roi_y_end_ratio"] = roi.get(
            "y_end_ratio", cfg["roi_y_end_ratio"]
        )
        for color, section in (("yellow", yellow), ("green", green)):
            for key in ("h_min", "h_max", "s_min", "s_max", "v_min", "v_max"):
                cfg[f"{color}_{key}"] = section.get(
                    key, cfg[f"{color}_{key}"]
                )
            cfg[f"{color}_prefer_bottom_connected"] = section.get(
                "prefer_bottom_connected",
                cfg[f"{color}_prefer_bottom_connected"],
            )
        for key in (
            "open_kernel_size",
            "close_kernel_size",
            "open_iterations",
            "close_iterations",
            "min_component_area",
            "bottom_band_pixels",
        ):
            cfg[key] = morphology.get(key, cfg[key])
        auto_mapping = {
            "sample_frames": "auto_sample_frames",
            "max_samples_per_color_per_frame":
                "auto_max_samples_per_color_per_frame",
            "low_percentile": "auto_low_percentile",
            "high_percentile": "auto_high_percentile",
            "hue_margin": "auto_hue_margin",
            "saturation_margin": "auto_saturation_margin",
            "value_margin": "auto_value_margin",
            "min_total_samples_per_color":
                "auto_min_total_samples_per_color",
        }
        for saved_name, runtime_name in auto_mapping.items():
            cfg[runtime_name] = auto.get(
                saved_name, cfg[runtime_name]
            )
        return ThresholdConfig.validate(cfg)

    def save_config(self) -> Dict[str, object]:
        """保存当前 profile 的 JSON 和人工可读 YAML。"""
        payload = self._config_payload()
        json_path, yaml_path = self.store.save(
            payload, self.profile_name, self.explicit_config_path
        )
        with self.state_lock:
            self.state.last_saved_path = str(json_path)
            self.state.last_error = ""
        return {
            "ok": True,
            "json_path": str(json_path),
            "yaml_path": str(yaml_path),
        }

    def load_config(self, require_exists: bool = True) -> Dict[str, object]:
        """从 JSON 加载配置；不存在时可静默保留内置值。"""
        path = self.store.load_path(
            self.profile_name, self.explicit_config_path
        )
        if not path.exists():
            if require_exists:
                raise APIError(f"配置文件不存在：{path}")
            return {"ok": True, "loaded": False, "path": str(path)}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise APIError("JSON 顶层必须是对象")
            validated = self._flat_config_from_payload(payload)
            loaded_profile = payload.get("profile_name", self.profile_name)
            loaded_profile = self.store.validate_profile(
                str(loaded_profile)
            )
        except (OSError, json.JSONDecodeError, APIError) as exc:
            with self.state_lock:
                self.state.last_error = f"加载配置失败：{exc}"
            raise APIError(f"加载配置失败：{exc}") from exc
        with self.cfg_lock:
            self.config = ThresholdConfig(validated)
        with self.state_lock:
            self.profile_name = loaded_profile
            self.state.profile_name = loaded_profile
            self.state.last_loaded_path = str(path)
            self.state.last_error = ""
        return {"ok": True, "loaded": True, "path": str(path)}

    def api_params(self) -> Dict[str, object]:
        """返回当前运行时参数和 GUI schema。"""
        cfg = self.cfg_snapshot()
        return {
            "ok": True,
            "params": {**cfg, "profile_name": self.profile_name},
            "schema": {
                name: list(schema)
                for name, schema in PARAMETER_SCHEMA.items()
            },
        }

    def api_status(self) -> Dict[str, object]:
        """返回图像、分割、自动采样的稳定状态。"""
        cfg = self.cfg_snapshot()
        with self.state_lock:
            status = self.state.status(cfg)
        status.update(self.auto_status())
        return status

    def pixel_info(self, x_text: str, y_text: str) -> Dict[str, object]:
        """读取最新原图指定像素的 BGR/HSV 和阈值命中状态。"""
        try:
            x = int(round(float(x_text)))
            y = int(round(float(y_text)))
        except ValueError as exc:
            raise APIError("像素坐标必须是数字") from exc
        with self.state_lock:
            if self.state.latest_bgr is None:
                raise APIError("尚未收到相机图像")
            image = self.state.latest_bgr.copy()
        height, width = image.shape[:2]
        if not 0 <= x < width or not 0 <= y < height:
            raise APIError(f"像素坐标超出图像范围：{width}x{height}")
        cfg = self.cfg_snapshot()
        bgr = [int(value) for value in image[y, x]]
        hsv_pixel = cv2.cvtColor(
            np.uint8([[bgr]]), cv2.COLOR_BGR2HSV
        )[0, 0]
        hsv = [int(value) for value in hsv_pixel]
        roi_start = int(height * float(cfg["roi_y_start_ratio"]))
        roi_end = int(height * float(cfg["roi_y_end_ratio"]))
        in_roi = roi_start <= y < roi_end

        def hits(color: str) -> bool:
            return in_roi and all(
                int(cfg[f"{color}_{channel}_min"]) <= hsv[index]
                <= int(cfg[f"{color}_{channel}_max"])
                for index, channel in enumerate(("h", "s", "v"))
            )

        return {
            "ok": True,
            "x": x,
            "y": y,
            "bgr": bgr,
            "hsv": hsv,
            "in_roi": in_roi,
            "hits_yellow": hits("yellow"),
            "hits_green": hits("green"),
        }

    def _start_web_server(self) -> None:
        """启动不阻塞 ROS 回调的 daemon HTTP 线程。"""
        web = ThresholdWebServer(self)
        self.http_server = ThreadingHTTPServer(
            (self.web_gui_host, self.web_gui_port), web.handler_class()
        )
        self.http_server.daemon_threads = True
        self.http_thread = threading.Thread(
            target=self.http_server.serve_forever,
            name="threshold-web-gui",
            daemon=True,
        )
        self.http_thread.start()
        self.get_logger().info(
            f"Web GUI: http://{self.web_gui_host}:{self.web_gui_port}"
        )
        if self.web_gui_host == "0.0.0.0":
            self.get_logger().warning(
                "Web GUI 无身份认证，请仅在可信机器人网络中使用。"
            )

    def shutdown(self) -> None:
        """关闭 Web 服务并唤醒所有等待图像的客户端。"""
        server = self.http_server
        self.http_server = None
        if server is not None:
            server.shutdown()
            server.server_close()
        with self.frame_condition:
            self.frame_condition.notify_all()


WEB_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport"
content="width=device-width,initial-scale=1"><title>黄绿 HSV 标定</title>
<style>
body{margin:0;background:#111;color:#eee;font:14px system-ui}
header{position:sticky;top:0;z-index:3;background:#222;padding:9px;display:flex;
gap:10px;align-items:center;flex-wrap:wrap}.warn{color:#ffb347}
button{padding:7px 11px}.layout{display:grid;grid-template-columns:minmax(360px,2fr)
minmax(320px,1fr);gap:10px;padding:10px}.card{background:#202020;padding:9px;
border-radius:6px;margin-bottom:9px}.raw{width:100%;height:auto;object-fit:contain;
image-rendering:auto;background:#181818}.thumbs{display:grid;
grid-template-columns:repeat(2,minmax(180px,1fr));gap:9px}.thumbs img{width:100%;
height:auto;object-fit:contain}.row{display:grid;grid-template-columns:1fr minmax(165px,1fr);
gap:6px;margin:5px 0}.ctrl{display:grid;grid-template-columns:1fr 72px;gap:5px}
input,select{width:100%;box-sizing:border-box;background:#333;color:#fff}
pre{white-space:pre-wrap;word-break:break-word}.ok{color:#74d680}.error{color:#ff7373}
@media(max-width:900px){.layout{grid-template-columns:1fr}.thumbs{grid-template-columns:1fr}}
</style></head><body>
<header><b>黄绿 HSV 标定</b><span id="topic"></span><span id="size">等待相机图像</span>
<span id="fps"></span><span id="auto"></span>
<button onclick="action('/api/save')">保存</button>
<button onclick="action('/api/load')">加载</button>
<button onclick="action('/api/defaults')">恢复默认</button>
<span class="warn">Web GUI 无身份认证，请仅在可信机器人网络中使用。</span></header>
<div class="layout"><section>
<div class="card"><h3>完整原始相机画面（点击查看 HSV）</h3>
<img id="raw" class="raw" src="/stream/raw" alt="等待相机图像"></div>
<div class="thumbs">
<div class="card"><b>Overlay / ROI / 接触预览</b><img src="/stream/overlay"></div>
<div class="card"><b>Yellow mask</b><img src="/stream/yellow"></div>
<div class="card"><b>Green mask</b><img src="/stream/green"></div>
<div class="card"><b>Unknown mask</b><img src="/stream/unknown"></div></div>
<pre id="status" class="card">等待相机图像</pre></section>
<aside><div class="card"><b>配置 Profile</b><div class="row">
<label>profile_name</label><input id="profile" value="usb_camera"
onchange="setv('profile_name',this.value)"></div></div>
<div class="card"><b>自动采样</b><p>
<button onclick="action('/api/auto/start')">开始</button>
<button onclick="action('/api/auto/stop')">停止</button>
<button onclick="action('/api/auto/reset')">清空</button>
<button onclick="action('/api/auto/apply')">应用建议</button></p>
<pre id="autoDetail"></pre></div>
<div class="card"><b>鼠标点 HSV</b><pre id="pixel">点击原图查看</pre></div>
<div id="controls"></div><pre id="message" class="card"></pre></aside></div>
<script>
let params={},schema={};
async function jsonFetch(url,opt){let r=await fetch(url,opt),x=await r.json();
if(!r.ok)throw Error(x.error||r.statusText);return x}
function field(name,value,s){let row=document.createElement('div');row.className='row';
let label=document.createElement('label');label.textContent=name;row.appendChild(label);
let kind=s[4],control;if(kind==='bool'){control=document.createElement('input');
control.type='checkbox';control.checked=!!value;control.onchange=()=>setv(name,control.checked)}
else{control=document.createElement('div');control.className='ctrl';
let range=document.createElement('input'),number=document.createElement('input');
range.type='range';number.type='number';[range,number].forEach(e=>{e.min=s[1];e.max=s[2];
e.step=s[3];e.value=value});range.oninput=()=>number.value=range.value;
number.oninput=()=>range.value=number.value;range.onchange=()=>setv(name,range.value);
number.onchange=()=>setv(name,number.value);control.append(range,number)}
row.appendChild(control);return row}
async function loadParams(){let x=await jsonFetch('/api/params');params=x.params;schema=x.schema;
document.getElementById('profile').value=params.profile_name;let groups={};
Object.keys(schema).forEach(n=>(groups[schema[n][0]]??=[]).push(n));
let root=document.getElementById('controls');root.innerHTML='';
Object.keys(groups).forEach(g=>{let box=document.createElement('div');box.className='card';
box.innerHTML='<b>'+g+'</b>';groups[g].forEach(n=>box.appendChild(field(n,params[n],schema[n])));
root.appendChild(box)})}
async function setv(n,v){try{let x=await jsonFetch('/api/set?name='+encodeURIComponent(n)+
'&value='+encodeURIComponent(v),{method:'POST'});show(JSON.stringify(x))}
catch(e){show(e.message,true)}await loadParams()}
async function action(url){try{let x=await jsonFetch(url,{method:'POST'});
show(JSON.stringify(x,null,2));await loadParams()}catch(e){show(e.message,true)}}
function show(t,error=false){let e=document.getElementById('message');e.textContent=t;
e.className='card '+(error?'error':'ok')}
async function tick(){try{let s=await jsonFetch('/api/status');
document.getElementById('topic').textContent=s.image_topic;
document.getElementById('size').textContent=s.image_received?s.frame_width+'×'+s.frame_height:
'等待相机图像';document.getElementById('fps').textContent=(s.fps||0).toFixed(1)+' FPS';
document.getElementById('auto').textContent=s.auto_message;
document.getElementById('status').textContent=JSON.stringify(s,null,2);
document.getElementById('autoDetail').textContent=JSON.stringify({sampled_frames:
s.auto_sampled_frames,requested_frames:s.auto_requested_frames,yellow_samples:
s.auto_yellow_sample_count,green_samples:s.auto_green_sample_count,yellow_median:
s.yellow_median_hsv,green_median:s.green_median_hsv,low_percentile:
s.auto_low_percentile,high_percentile:s.auto_high_percentile,yellow_percentile_low:
s.yellow_percentile_low_hsv,yellow_percentile_high:s.yellow_percentile_high_hsv,
green_percentile_low:s.green_percentile_low_hsv,green_percentile_high:
s.green_percentile_high_hsv,suggested_yellow:s.suggested_yellow,
suggested_green:s.suggested_green},null,2)}catch(e){}}
document.getElementById('raw').onclick=async e=>{let img=e.currentTarget,r=img.getBoundingClientRect();
let x=Math.floor((e.clientX-r.left)*img.naturalWidth/r.width);
let y=Math.floor((e.clientY-r.top)*img.naturalHeight/r.height);
try{let p=await jsonFetch('/api/pixel?x='+x+'&y='+y);
document.getElementById('pixel').textContent=JSON.stringify(p,null,2)}
catch(err){document.getElementById('pixel').textContent=err.message}};
loadParams();tick();setInterval(tick,500);
</script></body></html>"""


class ThresholdWebServer:
    """标准库 HTTP/MJPEG 接口。"""

    def __init__(self, node: GreenYellowThresholdNode) -> None:
        self.node = node

    def handler_class(self):
        """创建捕获当前节点的请求处理类。"""
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
                body = json.dumps(
                    value, ensure_ascii=False
                ).encode("utf-8")
                self.send_bytes(
                    status, body, "application/json; charset=utf-8"
                )

            def stream(self, stream_name: str) -> None:
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
                        frame = node.state.encoded_frames.get(stream_name)
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
                        stream_name = parsed.path.rsplit("/", 1)[-1]
                        if stream_name not in (
                            "raw", "overlay", "yellow", "green", "unknown"
                        ):
                            raise APIError("未知图像流", status=404)
                        self.stream(stream_name)
                    elif parsed.path == "/api/params":
                        self.send_json(node.api_params())
                    elif parsed.path == "/api/status":
                        self.send_json(node.api_status())
                    elif parsed.path == "/api/config":
                        self.send_json(node._config_payload())
                    elif parsed.path == "/api/set":
                        name = query.get("name", [""])[0]
                        value = query.get("value", [""])[0]
                        self.send_json(node.update_parameter(name, value))
                    elif parsed.path == "/api/auto/start":
                        self.send_json(node.auto_start())
                    elif parsed.path == "/api/auto/stop":
                        self.send_json(node.auto_stop())
                    elif parsed.path == "/api/auto/reset":
                        self.send_json(node.auto_reset())
                    elif parsed.path == "/api/auto/apply":
                        self.send_json(node.auto_apply())
                    elif parsed.path == "/api/save":
                        self.send_json(node.save_config())
                    elif parsed.path == "/api/load":
                        self.send_json(node.load_config(require_exists=True))
                    elif parsed.path == "/api/defaults":
                        self.send_json(node.restore_defaults())
                    elif parsed.path == "/api/pixel":
                        self.send_json(
                            node.pixel_info(
                                query.get("x", [""])[0],
                                query.get("y", [""])[0],
                            )
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
    """启动 ROS 节点；退出时只关闭 Web 服务。"""
    rclpy.init(args=args)
    node: Optional[GreenYellowThresholdNode] = None
    exit_code = 1
    try:
        node = GreenYellowThresholdNode()
        rclpy.spin(node)
        exit_code = 0
    except KeyboardInterrupt:
        exit_code = 0
    except Exception as exc:
        if node is not None:
            node.get_logger().error(f"标定工具异常：{exc}")
        else:
            print(f"标定工具启动失败：{exc}", flush=True)
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
