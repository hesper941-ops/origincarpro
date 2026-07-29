#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
固定相机逆透视（IPM）与米制坐标标定工具。

本程序只负责固定相机的逆透视与米制坐标标定，不包含道路中心线规划、路径规划和运动控制功能。

现场使用流程
--------------
1. 固定相机安装位置和 640×480 分辨率，标定完成后不得移动相机。
2. 将 0.50 m × 0.70 m 的长方形标定板完全平铺在车辆前方同一地面平面：
   0.50 m 短边沿车辆横向，0.70 m 长边与车辆前进方向平行；标定板不得翘曲。
3. 本版标定板内部不需要小方格，只在四个外角做清楚标记。四角必须全部进入画面，
   标定板尽量覆盖画面下部到中部，不要全部挤在画面上半部。
4. 实测标定板近边到车辆鸟瞰参考原点的距离，并填写横向中心偏移。
5. 浏览器打开 ``http://小车IP:8091``，点击“采集标定帧”，在后端保存的静态图中依次点击：
   P1 TL 远端左角、P2 TR 远端右角、P3 BL 近端左角、P4 BR 近端右角。
6. 角点和外框由后端绘制；本工具不支持拖动，点错后撤销，或一次性输入四点坐标。
7. 验证彩色鸟瞰图、车辆原点/中心轴及米制测距后保存配置。

为何使用 0.50 m × 0.70 m：它比 1 m 方板更适合当前 USB 相机的近场和中场，
可以更靠近车辆，同时 0.70 m 长度仍能提供足够透视差，0.50 m 宽度也接近赛道标称宽度。

未来 ``gxb_test/4_lane_geometry_planner.py`` 应读取
``gxb_test/config/usb_camera_ipm.json``，接口版本固定为 ``gxb_ipm_v1``。
"""

import copy
import json
import math
import os
import re
import threading
import time
import urllib.parse
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String


INTERFACE_VERSION = "gxb_ipm_v1"
FUTURE_PLANNER_FILE = "gxb_test/4_lane_geometry_planner.py"
DEFAULT_IMAGE_WIDTH = 640
DEFAULT_IMAGE_HEIGHT = 480
POINT_NAMES = ("P1 TL", "P2 TR", "P3 BL", "P4 BR")
POINT_KEYS = ("top_left", "top_right", "bottom_left", "bottom_right")
PROFILE_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class CalibrationError(ValueError):
    """可直接返回给 Web 页面的标定错误。"""


@dataclass
class IpmCalibrationConfig:
    """第三阶段所有稳定配置；标定板内部不使用小方格。"""

    image_topic: str = "/image_out"
    profile_name: str = "usb_camera"
    config_path: str = ""
    auto_load_config: bool = True
    calibration_board_width_m: float = 0.50
    calibration_board_length_m: float = 0.70
    calibration_grid_size_m: float = 0.10  # 仅保留米制参考，不绘制内部小方格。
    board_near_edge_distance_m: float = 0.20  # 待实车测量。
    board_center_lateral_offset_m: float = 0.0
    output_width_px: int = 600
    output_height_px: int = 800
    meter_per_pixel: float = 0.005
    vehicle_center_x_px: float = 300.0
    vehicle_origin_y_px: float = 799.0
    # 仅保存坐标语义，不查询 TF。必须与未来第四阶段采用的车辆参考原点一致。
    vehicle_reference_point_name: str = "base_link"
    expected_lane_width_m: float = 0.50  # 仅供未来阶段使用，待实车验证。
    single_boundary_center_offset_m: float = 0.25  # 仅供未来阶段使用。
    min_source_quad_area_px: float = 5000.0  # 待实车验证。
    homography_determinant_min_abs: float = 1.0e-9  # 待实车验证。
    subscribe_segmentation_topics: bool = True
    boundary_topic: str = "/gxb_test/final_boundary_mask"
    yellow_topic: str = "/gxb_test/yellow_mask_final"
    green_topic: str = "/gxb_test/green_mask_final"
    segmentation_status_topic: str = "/gxb_test/segmentation_status"
    publish_ipm_debug_topics: bool = True
    web_gui_enable: bool = True
    web_gui_host: str = "0.0.0.0"
    web_gui_port: int = 8091
    web_gui_jpeg_quality: int = 92
    web_gui_max_fps: float = 6.0

    def validate(self) -> None:
        """校验不会依赖相机帧的参数。"""
        if not PROFILE_PATTERN.fullmatch(self.profile_name):
            raise CalibrationError("profile_name 只允许字母、数字、下划线和短横线")
        if self.output_width_px <= 0 or self.output_height_px <= 0:
            raise CalibrationError("鸟瞰输出宽高必须大于零")
        if not (0.0001 <= self.meter_per_pixel <= 0.1):
            raise CalibrationError("meter_per_pixel 必须位于 0.0001 到 0.1")
        if self.calibration_board_width_m <= 0 or self.calibration_board_length_m <= 0:
            raise CalibrationError("标定板宽度和长度必须大于零")
        if self.calibration_grid_size_m <= 0:
            raise CalibrationError("米制参考间距必须大于零")
        if self.board_near_edge_distance_m < 0:
            raise CalibrationError("标定板近边距离不能为负数")
        if not (0 <= self.vehicle_center_x_px < self.output_width_px):
            raise CalibrationError("vehicle_center_x_px 超出鸟瞰图范围")
        if not (0 <= self.vehicle_origin_y_px < self.output_height_px):
            raise CalibrationError("vehicle_origin_y_px 超出鸟瞰图范围")
        if (
            not self.vehicle_reference_point_name
            or len(self.vehicle_reference_point_name) > 128
            or ".." in self.vehicle_reference_point_name
            or not re.fullmatch(
                r"[A-Za-z0-9_/]+", self.vehicle_reference_point_name
            )
        ):
            raise CalibrationError(
                "vehicle_reference_point_name 只能包含字母、数字、下划线和斜杠"
            )
        if not (1 <= self.web_gui_port <= 65535):
            raise CalibrationError("Web 端口必须位于 1 到 65535")
        if not (1 <= self.web_gui_jpeg_quality <= 100):
            raise CalibrationError("JPEG 质量必须位于 1 到 100")
        if self.web_gui_max_fps <= 0:
            raise CalibrationError("Web 最大帧率必须大于零")


@dataclass
class ValidationResult:
    """一次 Homography 校验结果。"""

    valid: bool = False
    reason: str = "等待选择四个角点"
    source_quad_area_px: float = 0.0
    homography_determinant: float = 0.0
    expected_board_width_px: float = 0.0
    expected_board_length_px: float = 0.0
    expected_reference_size_px: float = 0.0


@dataclass
class IpmRuntimeStatus:
    """供 Web 和 ROS 状态话题使用的运行状态。"""

    calibration_valid: bool = False
    calibration_reason: str = "等待选择四个角点"
    image_received: bool = False
    image_width: int = 0
    image_height: int = 0
    input_encoding: str = ""
    last_frame_time: float = 0.0
    boundary_mask_received: bool = False
    yellow_mask_received: bool = False
    green_mask_received: bool = False
    segmentation_status_received: bool = False
    selected_point_count: int = 0
    next_point_name: str = POINT_NAMES[0]
    fps: float = 0.0
    processing_time_ms: float = 0.0
    last_error: str = ""
    config_loaded: bool = False
    config_warning: str = ""


class ImageCodec:
    """不依赖图像桥接库的 ROS Image 与 OpenCV 数组转换。"""

    CHANNELS = {"bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4, "mono8": 1}

    @classmethod
    def to_bgr(cls, msg: Image) -> np.ndarray:
        encoding = str(msg.encoding).lower()
        if encoding not in cls.CHANNELS:
            raise CalibrationError(f"不支持图像编码: {msg.encoding}")
        width, height = int(msg.width), int(msg.height)
        channels = cls.CHANNELS[encoding]
        row_bytes = width * channels
        step = int(msg.step)
        if width <= 0 or height <= 0 or step < row_bytes:
            raise CalibrationError(
                f"非法图像尺寸或 step: {width}x{height}, step={step}, 至少需要 {row_bytes}"
            )
        raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        needed = step * height
        if raw.size < needed:
            raise CalibrationError(f"图像数据不足: {raw.size} < {needed}")
        packed = raw[:needed].reshape(height, step)[:, :row_bytes]
        if channels == 1:
            mono = packed.reshape(height, width)
            return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
        image = packed.reshape(height, width, channels)
        if encoding == "bgr8":
            return image.copy()
        if encoding == "rgb8":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if encoding == "bgra8":
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    @staticmethod
    def mono_from_message(msg: Image, expected_size: Tuple[int, int]) -> np.ndarray:
        if str(msg.encoding).lower() != "mono8":
            raise CalibrationError(f"mask 编码必须为 mono8，收到 {msg.encoding}")
        width, height = int(msg.width), int(msg.height)
        if (width, height) != expected_size:
            raise CalibrationError(
                f"mask 分辨率 {width}x{height} 与原图 {expected_size[0]}x{expected_size[1]} 不一致"
            )
        step = int(msg.step)
        if step < width:
            raise CalibrationError("mask step 小于宽度")
        raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        if raw.size < step * height:
            raise CalibrationError("mask 数据长度不足")
        mask = raw[: step * height].reshape(height, step)[:, :width].copy()
        return np.where(mask > 127, 255, 0).astype(np.uint8)

    @staticmethod
    def to_message(array: np.ndarray, encoding: str, source: Optional[Image]) -> Image:
        output = Image()
        if source is not None:
            output.header = source.header
        contiguous = np.ascontiguousarray(array)
        output.height, output.width = contiguous.shape[:2]
        output.encoding = encoding
        output.is_bigendian = 0
        output.step = output.width * (1 if contiguous.ndim == 2 else contiguous.shape[2])
        output.data = contiguous.tobytes()
        return output


class CalibrationPointManager:
    """维护严格按 TL、TR、BL、BR 排列的四个原图点。"""

    def __init__(self) -> None:
        self.points: List[List[float]] = []

    def set_point(
        self, index: int, x: float, y: float, image_width: int, image_height: int
    ) -> None:
        if index < 0 or index >= 4:
            raise CalibrationError("点序号必须位于 0 到 3")
        if not (0 <= x < image_width and 0 <= y < image_height):
            raise CalibrationError(f"{POINT_NAMES[index]} 超出原图范围")
        point = [float(x), float(y)]
        if index > len(self.points):
            raise CalibrationError(f"请先选择 {POINT_NAMES[len(self.points)]}")
        if index == len(self.points):
            self.points.append(point)
        else:
            self.points[index] = point

    def undo(self) -> None:
        if self.points:
            self.points.pop()

    def clear(self) -> None:
        self.points.clear()

    def load(self, points: Sequence[Sequence[float]]) -> None:
        if len(points) != 4:
            raise CalibrationError("配置中的 source_points 必须正好包含四点")
        self.points = [[float(p[0]), float(p[1])] for p in points]

    def next_name(self) -> str:
        return POINT_NAMES[len(self.points)] if len(self.points) < 4 else "四点已完整"


class DestinationPointCalculator:
    """根据真实米制尺寸动态计算鸟瞰目标四点。"""

    @staticmethod
    def calculate(config: IpmCalibrationConfig) -> np.ndarray:
        scale = config.meter_per_pixel
        half_width = config.calibration_board_width_m / scale / 2.0
        length = config.calibration_board_length_m / scale
        near_distance = config.board_near_edge_distance_m / scale
        center_offset = config.board_center_lateral_offset_m / scale
        center_x = config.vehicle_center_x_px - center_offset
        bottom_y = config.vehicle_origin_y_px - near_distance
        top_y = bottom_y - length
        return np.asarray(
            [
                [center_x - half_width, top_y],
                [center_x + half_width, top_y],
                [center_x - half_width, bottom_y],
                [center_x + half_width, bottom_y],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def validate(points: np.ndarray, config: IpmCalibrationConfig) -> None:
        if not np.all(np.isfinite(points)):
            raise CalibrationError("目的点包含 NaN 或 Inf")
        for index, (x, y) in enumerate(points):
            if not (0 <= x < config.output_width_px and 0 <= y < config.output_height_px):
                raise CalibrationError(
                    f"目的点 {POINT_NAMES[index]}=({x:.1f},{y:.1f}) 越界；请调整输出尺寸、"
                    "meter_per_pixel、近边距离、车辆原点/中心轴或横向偏移"
                )
        if not (points[0, 1] < points[2, 1]):
            raise CalibrationError("目的点远边必须位于近边上方")


class HomographyValidator:
    """执行四点拓扑、分辨率和矩阵数值校验。"""

    @staticmethod
    def _cross(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float(np.cross(b - a, c - a))

    @classmethod
    def _segments_intersect(
        cls, a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
    ) -> bool:
        return cls._cross(a, b, c) * cls._cross(a, b, d) < 0 and cls._cross(
            c, d, a
        ) * cls._cross(c, d, b) < 0

    @classmethod
    def compute(
        cls,
        source: Sequence[Sequence[float]],
        destination: np.ndarray,
        config: IpmCalibrationConfig,
        image_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, ValidationResult]:
        if len(source) != 4:
            raise CalibrationError("必须依次选择 TL、TR、BL、BR 四点")
        config.validate()
        DestinationPointCalculator.validate(destination, config)
        src = np.asarray(source, dtype=np.float32)
        width, height = image_size
        if (width, height) != (DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT):
            raise CalibrationError(
                f"输入分辨率必须固定为 {DEFAULT_IMAGE_WIDTH}x{DEFAULT_IMAGE_HEIGHT}，"
                f"当前为 {width}x{height}"
            )
        if not np.all(np.isfinite(src)):
            raise CalibrationError("源点包含 NaN 或 Inf")
        for index, (x, y) in enumerate(src):
            if not (0 <= x < width and 0 <= y < height):
                raise CalibrationError(f"{POINT_NAMES[index]} 超出原图范围")
        if min(np.linalg.norm(src[i] - src[j]) for i in range(4) for j in range(i)) < 2:
            raise CalibrationError("四个源点不能重合")
        # TL/TR/BL/BR 不是环形顺序，转换成 TL/TR/BR/BL 后检查多边形。
        polygon = src[[0, 1, 3, 2]]
        area = abs(float(cv2.contourArea(polygon)))
        if area < config.min_source_quad_area_px:
            raise CalibrationError(
                f"源四边形面积 {area:.1f}px² 小于阈值 {config.min_source_quad_area_px:.1f}px²"
            )
        if cls._segments_intersect(src[0], src[1], src[2], src[3]) or cls._segments_intersect(
            src[0], src[2], src[1], src[3]
        ):
            raise CalibrationError("源四边形存在自交")
        if not (src[0, 0] < src[1, 0] and src[2, 0] < src[3, 0]):
            raise CalibrationError("左右顺序错误：TL/BL 必须分别位于 TR/BR 左侧")
        if not (
            max(src[0, 1], src[1, 1]) < min(src[2, 1], src[3, 1])
        ):
            raise CalibrationError("远近顺序错误：TL/TR 必须位于 BL/BR 上方")
        matrix = cv2.getPerspectiveTransform(src, destination.astype(np.float32))
        if not np.all(np.isfinite(matrix)):
            raise CalibrationError("Homography 包含 NaN 或 Inf")
        determinant = float(np.linalg.det(matrix))
        if abs(determinant) < config.homography_determinant_min_abs:
            raise CalibrationError(f"Homography 行列式过小: {determinant:.3e}")
        try:
            inverse = np.linalg.inv(matrix)
        except np.linalg.LinAlgError as exc:
            raise CalibrationError("Homography 无法求逆") from exc
        if not np.all(np.isfinite(inverse)):
            raise CalibrationError("逆 Homography 包含 NaN 或 Inf")
        result = ValidationResult(
            valid=True,
            reason="valid",
            source_quad_area_px=area,
            homography_determinant=determinant,
            expected_board_width_px=config.calibration_board_width_m
            / config.meter_per_pixel,
            expected_board_length_px=config.calibration_board_length_m
            / config.meter_per_pixel,
            expected_reference_size_px=config.calibration_grid_size_m
            / config.meter_per_pixel,
        )
        return matrix, inverse, result


class IpmTransformer:
    """彩色图与二值 mask 使用不同插值规则进行逆透视。"""

    @staticmethod
    def color(image: np.ndarray, matrix: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        return cv2.warpPerspective(
            image, matrix, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )

    @staticmethod
    def mask(mask: np.ndarray, matrix: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        warped = cv2.warpPerspective(
            mask, matrix, size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
        )
        return np.where(warped > 127, 255, 0).astype(np.uint8)


class IpmConfigManager:
    """安全保存/加载第四阶段正式使用的 JSON 和人工查看 YAML。"""

    def __init__(self, script_path: Path) -> None:
        self.config_dir = script_path.resolve().parent / "config"

    @staticmethod
    def validate_profile(profile: str) -> None:
        if not PROFILE_PATTERN.fullmatch(profile):
            raise CalibrationError("非法 profile_name，禁止路径字符和路径穿越")

    def paths(self, config: IpmCalibrationConfig) -> Tuple[Path, Path]:
        self.validate_profile(config.profile_name)
        if config.config_path:
            path = Path(config.config_path)
            if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".json":
                raise CalibrationError("config_path 必须是安全的仓库内相对 JSON 路径")
            resolved = (Path.cwd() / path).resolve()
            repository = Path.cwd().resolve()
            if repository not in resolved.parents:
                raise CalibrationError("config_path 超出仓库范围")
            json_path = resolved
        else:
            json_path = self.config_dir / f"{config.profile_name}_ipm.json"
        return json_path, json_path.with_suffix(".yaml")

    @staticmethod
    def _point_dict(points: Sequence[Sequence[float]]) -> Dict[str, List[float]]:
        return {
            key: [round(float(points[i][0]), 6), round(float(points[i][1]), 6)]
            for i, key in enumerate(POINT_KEYS)
        }

    def make_document(
        self,
        config: IpmCalibrationConfig,
        source: Sequence[Sequence[float]],
        destination: np.ndarray,
        matrix: np.ndarray,
        inverse: np.ndarray,
        validation: ValidationResult,
        image_size: Tuple[int, int],
    ) -> Dict[str, Any]:
        validation_document = asdict(validation)
        # 保留接口字段名；这里的 0.10 m 仅表示米制参考长度，不代表标定板有内部网格。
        validation_document["expected_grid_size_px"] = (
            validation.expected_reference_size_px
        )
        return {
            "version": INTERFACE_VERSION,
            "profile_name": config.profile_name,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "image_topic": config.image_topic,
            "image_width": image_size[0],
            "image_height": image_size[1],
            "coordinate_convention": {
                "birdview_pixel_x": "right_positive",
                "birdview_pixel_y": "down_positive",
                "vehicle_forward": "up_in_birdview",
                "vehicle_left": "left_in_birdview",
                "vehicle_origin": "configured_reference_point",
                "vehicle_reference_point_name": config.vehicle_reference_point_name,
                "forward_formula": "(vehicle_origin_y_px - birdview_y_px) * meter_per_pixel",
                "left_formula": "(vehicle_center_x_px - birdview_x_px) * meter_per_pixel",
            },
            "calibration_board": {
                "width_m": config.calibration_board_width_m,
                "length_m": config.calibration_board_length_m,
                "grid_size_m": config.calibration_grid_size_m,
                "grid_columns": 0,
                "grid_rows": 0,
                "internal_grid_used": False,
                "corner_marks_only": True,
                "near_edge_distance_m": config.board_near_edge_distance_m,
                "center_lateral_offset_m": config.board_center_lateral_offset_m,
            },
            "source_points": self._point_dict(source),
            "destination_points": self._point_dict(destination),
            "output": {
                "width_px": config.output_width_px,
                "height_px": config.output_height_px,
                "meter_per_pixel": config.meter_per_pixel,
                "vehicle_center_x_px": config.vehicle_center_x_px,
                "vehicle_origin_y_px": config.vehicle_origin_y_px,
            },
            "road_geometry_defaults": {
                "expected_lane_width_m": config.expected_lane_width_m,
                "single_boundary_center_offset_m": config.single_boundary_center_offset_m,
            },
            "homography_matrix": matrix.tolist(),
            "inverse_homography_matrix": inverse.tolist(),
            "validation": validation_document,
        }

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

    @staticmethod
    def _yaml_scalar(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return "null"
        if isinstance(value, str):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    @classmethod
    def _yaml(cls, value: Any, indent: int = 0) -> str:
        lines: List[str] = []
        prefix = " " * indent
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.append(cls._yaml(item, indent + 2))
                else:
                    lines.append(f"{prefix}{key}: {cls._yaml_scalar(item)}")
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}-")
                    lines.append(cls._yaml(item, indent + 2))
                else:
                    lines.append(f"{prefix}- {cls._yaml_scalar(item)}")
        return "\n".join(lines)

    def save(self, document: Dict[str, Any], config: IpmCalibrationConfig) -> Tuple[Path, Path]:
        json_path, yaml_path = self.paths(config)
        self._atomic_write(json_path, json.dumps(document, ensure_ascii=False, indent=2) + "\n")
        self._atomic_write(yaml_path, self._yaml(document) + "\n")
        return json_path, yaml_path

    def load(self, config: IpmCalibrationConfig) -> Tuple[Path, Dict[str, Any]]:
        json_path, _ = self.paths(config)
        with json_path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        if document.get("version") != INTERFACE_VERSION:
            raise CalibrationError(f"配置版本必须为 {INTERFACE_VERSION}")
        return json_path, document


class WebFrameStore:
    """只保存每种预览的最新 JPEG，慢客户端不会积压帧。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frames: Dict[str, bytes] = {}
        self.sequence: Dict[str, int] = {}

    def update(self, name: str, image: np.ndarray, quality: int) -> None:
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return
        with self.lock:
            self.frames[name] = encoded.tobytes()
            self.sequence[name] = self.sequence.get(name, 0) + 1

    def get(self, name: str) -> Tuple[Optional[bytes], int]:
        with self.lock:
            return self.frames.get(name), self.sequence.get(name, 0)


WEB_PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>固定相机 IPM 静态帧标定</title>
<style>
body{margin:0;background:#10151d;color:#e7edf5;font:14px system-ui}.top{padding:12px 18px;background:#172131;position:sticky;top:0;z-index:2}
h1{margin:0 0 6px;font-size:21px}.warn{color:#ffcc66}.bad{color:#ff6b6b}.good{color:#6fe29b}
.page{padding:12px}.row{display:grid;grid-template-columns:minmax(0,2fr) minmax(320px,1fr);gap:12px;margin-bottom:12px}
.panel{background:#182231;border:1px solid #2b3a50;border-radius:8px;padding:12px;min-width:0}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:10px}.view h3{margin:4px 0}
img{width:100%;height:auto;display:block;background:#05080c}.clickable{cursor:crosshair}
.buttons{display:flex;flex-wrap:wrap;gap:7px;margin:9px 0}button{padding:7px 10px;background:#31577f;color:white;border:0;border-radius:5px;cursor:pointer}
button.danger{background:#893c49}label{display:grid;grid-template-columns:1fr 120px;gap:8px;margin:5px 0}
input{background:#0f1722;color:#eef;border:1px solid #40516a;border-radius:4px;padding:5px}
.points{display:grid;grid-template-columns:repeat(2,1fr);gap:6px}.small{font-size:12px;color:#aab8ca}
pre{white-space:pre-wrap;background:#0d131d;padding:8px;border-radius:5px;max-height:300px;overflow:auto}
@media(max-width:1000px){.row,.pair{grid-template-columns:1fr}}
</style></head><body>
<div class="top"><h1>第三阶段：固定相机 IPM 静态帧标定（gxb_ipm_v1）</h1>
<div id="summary">等待状态……</div>
<div class="warn">请先点击“采集标定帧”，随后只在下方静态图片中依次点击 TL、TR、BL、BR。实时图仅用于观察。</div></div>
<div class="page">
<section class="panel row"><div class="view"><h3>实时相机预览（不可点击）</h3><img id="liveImage" src="/stream/raw"></div>
<div class="view"><h3>静态鸟瞰结果</h3><img id="birdviewImage" src="/image/calibration_birdview.jpg"></div></section>
<section class="row"><div class="panel"><h3>静态标定图（依次点击四角）</h3>
<img id="calibrationImage" class="clickable" src="/image/calibration_annotated.jpg">
<p id="next" class="warn"></p><div class="buttons">
<button onclick="captureFrame()">采集标定帧</button><button onclick="captureFrame()">重新采集</button>
<button onclick="post('/api/undo_point')">撤销上一个点</button>
<button class="danger" onclick="post('/api/clear_points')">清空四点</button>
<button onclick="applyPoints()">应用手工坐标</button><button onclick="post('/api/recompute')">重新计算</button>
<button onclick="post('/api/validate')">验证标定</button><button onclick="post('/api/save')">保存配置</button>
<button onclick="post('/api/load')">加载配置</button><button onclick="post('/api/reset_defaults')">恢复默认参数</button>
</div><h3>四点原图坐标</h3><div id="pointInputs"></div></div>
<aside class="panel"><h3>参数</h3><div id="params"></div><pre id="status"></pre></aside></section>
<section class="panel"><h3>鸟瞰测距</h3><p>输入鸟瞰像素坐标测量未参与四角强制映射的已知距离。</p>
<div class="points"><input id="mx1" placeholder="x1"><input id="my1" placeholder="y1"><input id="mx2" placeholder="x2"><input id="my2" placeholder="y2"></div>
<div class="buttons"><button onclick="measure()">测量</button><button onclick="post('/api/clear_measurement')">清除测量</button><span id="measure"></span></div></section>
<details id="maskPanel" class="panel"><summary>第二阶段 mask 鸟瞰预览（可选）</summary><div class="pair">
<div><h3>原始 boundary</h3><img src="/stream/boundary_raw"></div><div><h3>鸟瞰 boundary</h3><img src="/stream/boundary_birdview"></div>
<div><h3>鸟瞰 yellow</h3><img src="/stream/yellow_birdview"></div><div><h3>鸟瞰 green</h3><img src="/stream/green_birdview"></div></div>
<p class="small">第二阶段不是四点标定的必要条件。</p></details>
<section class="panel"><h3>摆放与验收</h3><ol>
<li>相机固定为 640×480；0.50 m × 0.70 m 标定板完全平铺，内部没有实体网格，只标注四个外角。</li>
<li>短边沿车辆横向，长边沿前进方向；核对车辆实际 TF，近边距离从未来阶段使用的同一个车辆参考原点测量。</li>
<li>四个目标角被强制映射为 100×140 px，只能验证目的点和米像素比例，不能单独证明标定精度。</li>
<li>使用未参与四点计算的边缘中点标记、另一条已知长度或近/中场固定 0.50 m 赛道宽度进行人工验证。</li>
</ol></section></div>
<script>
const fields=['calibration_board_width_m','calibration_board_length_m','calibration_grid_size_m',
'board_near_edge_distance_m','board_center_lateral_offset_m','output_width_px','output_height_px',
'meter_per_pixel','vehicle_center_x_px','vehicle_origin_y_px','vehicle_reference_point_name',
'expected_lane_width_m','single_boundary_center_offset_m'];
let state={};
function el(id){return document.getElementById(id)}
function build(){
 el('params').innerHTML=fields.map(k=>`<label>${k}<input id="p_${k}"></label>`).join('');
 el('pointInputs').innerHTML=['P1 TL','P2 TR','P3 BL','P4 BR'].map((n,i)=>`<label>${n}<span><input id="x${i}" style="width:65px"> <input id="y${i}" style="width:65px"></span></label>`).join('');
 fields.forEach(k=>el('p_'+k).onchange=()=>post('/api/set',{[k]:el('p_'+k).value}));
}
function refreshStaticImages(){
 const stamp=Date.now();
 el('calibrationImage').src=`/image/calibration_annotated.jpg?t=${stamp}`;
 el('birdviewImage').src=`/image/calibration_birdview.jpg?t=${stamp}`;
}
async function post(url,data={}){
 const response=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
 const result=await response.json();if(!response.ok){await refresh(true);alert(result.error||'操作失败');throw new Error(result.error||'操作失败')}
 await refresh(true);return result;
}
async function captureFrame(){await post('/api/capture_calibration_frame')}
async function refresh(updateImages=false){
 try{
  state=await (await fetch('/api/status',{cache:'no-store'})).json();
  const cfg=await (await fetch('/api/config',{cache:'no-store'})).json();
  fields.forEach(k=>{if(document.activeElement!==el('p_'+k))el('p_'+k).value=cfg[k]});
  for(let i=0;i<4;i++){const p=(state.source_points||[])[i];if(document.activeElement!==el('x'+i))el('x'+i).value=p?p[0].toFixed(1):'';if(document.activeElement!==el('y'+i))el('y'+i).value=p?p[1].toFixed(1):''}
  el('summary').innerHTML=`image_received=${state.image_received} | calibration_frame_captured=${state.calibration_frame_captured} | sequence=${state.calibration_frame_sequence} | points=${state.selected_point_count} | <span class="${state.calibration_valid?'good':'bad'}">${state.calibration_valid?'标定有效':state.calibration_reason}</span>`;
  el('next').textContent=state.calibration_frame_captured?'当前等待点击：'+state.next_point_name:'请先采集标定帧';
  el('status').textContent=JSON.stringify(state,null,2);el('maskPanel').style.display=cfg.subscribe_segmentation_topics?'block':'none';
  if(updateImages)refreshStaticImages();
 }catch(error){el('summary').textContent='状态读取失败: '+error}
}
el('calibrationImage').addEventListener('click',async event=>{
 if(!state.calibration_frame_captured){alert('请先采集标定帧');return}
 const image=event.currentTarget,rect=image.getBoundingClientRect();
 if(!image.naturalWidth||rect.width<=0||rect.height<=0)return;
 const x=Math.max(0,Math.min(image.naturalWidth-1,Math.round((event.clientX-rect.left)*image.naturalWidth/rect.width)));
 const y=Math.max(0,Math.min(image.naturalHeight-1,Math.round((event.clientY-rect.top)*image.naturalHeight/rect.height)));
 await post('/api/add_point',{x:x,y:y});
});
async function applyPoints(){
 const points=[];for(let i=0;i<4;i++)points.push([Number(el('x'+i).value),Number(el('y'+i).value)]);
 await post('/api/set_points',{points:points});
}
async function measure(){const result=await post('/api/measure_point',{x1:+el('mx1').value,y1:+el('my1').value,x2:+el('mx2').value,y2:+el('my2').value});el('measure').textContent=`直线 ${result.distance_m.toFixed(4)} m；横向 ${result.lateral_distance_m.toFixed(4)} m；纵向 ${result.forward_distance_m.toFixed(4)} m`}
build();setInterval(()=>refresh(false),1000);refresh(true);
</script></body></html>"""


class IpmCalibrationNode(Node):
    """ROS、标定状态、Web 操作和预览发布的协调节点。"""

    CONFIG_FIELDS = {item.name for item in fields(IpmCalibrationConfig)}
    WEB_EDITABLE_FIELDS = {
        "calibration_board_width_m",
        "calibration_board_length_m",
        "calibration_grid_size_m",
        "board_near_edge_distance_m",
        "board_center_lateral_offset_m",
        "output_width_px",
        "output_height_px",
        "meter_per_pixel",
        "vehicle_center_x_px",
        "vehicle_origin_y_px",
        "vehicle_reference_point_name",
        "expected_lane_width_m",
        "single_boundary_center_offset_m",
    }

    def __init__(self) -> None:
        super().__init__("gxb_ipm_calibration")
        self.lock = threading.RLock()
        defaults = IpmCalibrationConfig()
        for name in self.CONFIG_FIELDS:
            self.declare_parameter(name, getattr(defaults, name))
        self.config = IpmCalibrationConfig(
            **{name: self.get_parameter(name).value for name in self.CONFIG_FIELDS}
        )
        self.config.validate()
        self.defaults = copy.deepcopy(self.config)
        self.status = IpmRuntimeStatus()
        self.points = CalibrationPointManager()
        self.validation = ValidationResult()
        self.matrix: Optional[np.ndarray] = None
        self.inverse_matrix: Optional[np.ndarray] = None
        self.latest_image: Optional[np.ndarray] = None
        self.latest_image_message: Optional[Image] = None
        # 静态标定状态与持续更新的实时相机缓存严格分离。
        self.calibration_frame_bgr: Optional[np.ndarray] = None
        self.calibration_frame_header: Optional[Any] = None
        self.calibration_frame_captured = False
        self.calibration_frame_sequence = 0
        self.calibration_frame_captured_at = ""
        self.calibration_masks: Dict[str, Optional[np.ndarray]] = {
            "boundary": None,
            "yellow": None,
            "green": None,
        }
        self.calibration_mask_messages: Dict[str, Optional[Image]] = {
            "boundary": None,
            "yellow": None,
            "green": None,
        }
        self.masks: Dict[str, Optional[np.ndarray]] = {
            "boundary": None,
            "yellow": None,
            "green": None,
        }
        self.mask_messages: Dict[str, Optional[Image]] = {
            "boundary": None,
            "yellow": None,
            "green": None,
        }
        self.measurement: Optional[Dict[str, float]] = None
        self.frames = WebFrameStore()
        self.config_manager = IpmConfigManager(Path(__file__))
        self.web_server: Optional[ThreadingHTTPServer] = None
        self.web_thread: Optional[threading.Thread] = None
        self.last_encode_time = 0.0
        self.frame_times: List[float] = []
        self._initialize_placeholders()
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(Image, self.config.image_topic, self._on_image, qos)
        if self.config.subscribe_segmentation_topics:
            self.create_subscription(
                Image, self.config.boundary_topic, lambda msg: self._on_mask("boundary", msg), qos
            )
            self.create_subscription(
                Image, self.config.yellow_topic, lambda msg: self._on_mask("yellow", msg), qos
            )
            self.create_subscription(
                Image, self.config.green_topic, lambda msg: self._on_mask("green", msg), qos
            )
            self.create_subscription(String, self.config.segmentation_status_topic, self._on_segmentation_status, qos)
        self.publishers: Dict[str, Any] = {}
        if self.config.publish_ipm_debug_topics:
            self.publishers = {
                "raw": self.create_publisher(Image, "/gxb_test/ipm/raw_birdview", 1),
                "boundary": self.create_publisher(Image, "/gxb_test/ipm/boundary_birdview", 1),
                "yellow": self.create_publisher(Image, "/gxb_test/ipm/yellow_birdview", 1),
                "green": self.create_publisher(Image, "/gxb_test/ipm/green_birdview", 1),
                "status": self.create_publisher(String, "/gxb_test/ipm/status", 1),
            }
        self.create_timer(0.5, self._publish_status)
        if self.config.auto_load_config:
            try:
                self._load_config(missing_ok=True)
            except Exception as exc:
                self.status.last_error = f"自动加载配置失败: {exc}"
        if self.config.web_gui_enable:
            self._start_web()
        self.get_logger().info(
            f"IPM 标定工具启动: image_topic={self.config.image_topic}, "
            f"web=http://{self.config.web_gui_host}:{self.config.web_gui_port}, "
            f"profile={self.config.profile_name}, interface={INTERFACE_VERSION}"
        )

    def _display_image(self) -> Optional[np.ndarray]:
        """所有标定计算只使用人工采集的静态帧。"""
        return self.calibration_frame_bgr

    def _display_image_message(self) -> Optional[Image]:
        if self.calibration_frame_header is None:
            return None
        snapshot = Image()
        snapshot.header = copy.deepcopy(self.calibration_frame_header)
        return snapshot

    def _display_masks(self) -> Dict[str, Optional[np.ndarray]]:
        return self.calibration_masks

    def _display_mask_messages(self) -> Dict[str, Optional[Image]]:
        return self.calibration_mask_messages

    @staticmethod
    def _header_snapshot(message: Optional[Image]) -> Optional[Image]:
        """只复制发布所需的消息头，避免静态快照引用后续可变状态。"""
        if message is None:
            return None
        snapshot = Image()
        snapshot.header = copy.deepcopy(message.header)
        return snapshot

    def _initialize_placeholders(self) -> None:
        """让无相机、无第二阶段时 Web 流仍能给出明确文字。"""
        raw = np.zeros((DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH, 3), dtype=np.uint8)
        cv2.putText(
            raw,
            "Waiting for /image_out",
            (135, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        bird = np.zeros((800, 600, 3), dtype=np.uint8)
        cv2.putText(
            bird,
            "Waiting for calibration",
            (110, 400),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        mask = np.zeros((360, 600), dtype=np.uint8)
        cv2.putText(
            mask,
            "Stage 2 output not received",
            (70, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            255,
            2,
            cv2.LINE_AA,
        )
        self.frames.update("live_raw", raw, 90)
        self.frames.update("calibration_frame", raw, 90)
        self.frames.update("calibration_annotated", raw, 90)
        self.frames.update("calibration_birdview", bird, 90)
        for name in (
            "boundary_raw",
            "boundary_birdview",
            "yellow_birdview",
            "green_birdview",
        ):
            self.frames.update(name, mask, 90)

    def _set_validation_error(self, message: str) -> None:
        self.matrix = None
        self.inverse_matrix = None
        self.validation = ValidationResult(valid=False, reason=message)
        self.status.calibration_valid = False
        self.status.calibration_reason = message

    def _recompute(self) -> ValidationResult:
        image = self._display_image()
        if image is None:
            raise CalibrationError("尚未采集静态标定帧")
        try:
            destination = DestinationPointCalculator.calculate(self.config)
            matrix, inverse, validation = HomographyValidator.compute(
                self.points.points,
                destination,
                self.config,
                (image.shape[1], image.shape[0]),
            )
        except Exception as exc:
            self._set_validation_error(str(exc))
            raise
        self.matrix, self.inverse_matrix, self.validation = matrix, inverse, validation
        self.status.calibration_valid = True
        self.status.calibration_reason = validation.reason
        self._refresh_static_previews()
        return validation

    def _on_image(self, msg: Image) -> None:
        start = time.perf_counter()
        try:
            bgr = ImageCodec.to_bgr(msg)
        except Exception as exc:
            with self.lock:
                self.status.last_error = f"图像解码失败: {exc}"
            return
        now = time.monotonic()
        with self.lock:
            self.latest_image = bgr
            self.latest_image_message = msg
            self.status.image_received = True
            self.status.image_width = bgr.shape[1]
            self.status.image_height = bgr.shape[0]
            self.status.input_encoding = str(msg.encoding)
            self.status.last_frame_time = now
            self.frame_times = [stamp for stamp in self.frame_times if now - stamp <= 1.0]
            self.frame_times.append(now)
            self.status.fps = float(len(self.frame_times))
            self.status.processing_time_ms = (time.perf_counter() - start) * 1000.0
            self.status.last_error = ""
            if (
                self.matrix is not None
                and (bgr.shape[1], bgr.shape[0])
                != (DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT)
            ):
                self._set_validation_error(
                    f"输入分辨率必须为 {DEFAULT_IMAGE_WIDTH}x{DEFAULT_IMAGE_HEIGHT}"
                )
        # 实时回调只更新观察用 MJPEG；不执行 Homography 或静态标注。
        if now - self.last_encode_time >= 1.0 / self.config.web_gui_max_fps:
            self.frames.update(
                "live_raw", bgr, self.config.web_gui_jpeg_quality
            )
            self.last_encode_time = now

    def _on_mask(self, name: str, msg: Image) -> None:
        with self.lock:
            expected = (self.status.image_width, self.status.image_height)
            if expected[0] <= 0:
                self.status.last_error = f"{name} mask 已到达，但尚未收到原图"
                return
            try:
                mask = ImageCodec.mono_from_message(msg, expected)
            except Exception as exc:
                self.status.last_error = f"{name} mask 被拒绝: {exc}"
                return
            source_header = (
                getattr(self.latest_image_message, "header", None)
                if self.latest_image_message is not None
                else None
            )
            mask_header = getattr(msg, "header", None)
            source_frame = str(getattr(source_header, "frame_id", ""))
            mask_frame = str(getattr(mask_header, "frame_id", ""))
            if source_frame and mask_frame and source_frame != mask_frame:
                self.status.last_error = (
                    f"{name} mask frame_id={mask_frame} 与原图 frame_id={source_frame} 不一致"
                )
                return
            self.masks[name] = mask
            self.mask_messages[name] = msg
            setattr(self.status, f"{name}_mask_received", True)
        # 实时 mask 只更新最新缓存；静态标定预览在采集时一次性快照。

    def _on_segmentation_status(self, _msg: String) -> None:
        with self.lock:
            self.status.segmentation_status_received = True

    def _annotate_calibration_frame(self, image: np.ndarray) -> np.ndarray:
        """在静态帧副本上由后端绘制角点，绝不修改原始快照。"""
        annotated = image.copy()
        points = np.asarray(self.points.points, dtype=np.int32)
        if len(points) >= 2:
            if len(points) == 4:
                order = points[[0, 1, 3, 2]]
                cv2.polylines(
                    annotated, [order], True, (255, 255, 255), 2, cv2.LINE_AA
                )
            else:
                cv2.polylines(
                    annotated, [points], False, (255, 255, 255), 2, cv2.LINE_AA
                )
        colors = (
            (0, 0, 255),      # P1 TL 红
            (0, 255, 0),      # P2 TR 绿
            (255, 0, 0),      # P3 BL 蓝
            (255, 0, 255),    # P4 BR 紫红
        )
        for index, point in enumerate(points):
            location = (int(point[0]), int(point[1]))
            cv2.circle(annotated, location, 7, colors[index], -1, cv2.LINE_AA)
            label_position = (location[0] + 10, max(18, location[1] - 8))
            # 黑色描边保证标签在浅色或高亮背景上仍清楚。
            cv2.putText(
                annotated,
                POINT_NAMES[index],
                label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                POINT_NAMES[index],
                label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                colors[index],
                2,
                cv2.LINE_AA,
            )
        progress_text = (
            f"Next: {self.points.next_name()}"
            if len(points) < 4
            else "Four points complete"
        )
        cv2.putText(
            annotated,
            progress_text,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            progress_text,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    def _annotate_birdview(self, image: np.ndarray) -> np.ndarray:
        """仅画坐标轴、车辆原点和标定板外框；不画内部小方格。"""
        overlay = image.copy()
        cx = int(round(self.config.vehicle_center_x_px))
        oy = int(round(self.config.vehicle_origin_y_px))
        cv2.line(overlay, (cx, 0), (cx, overlay.shape[0] - 1), (255, 130, 0), 1)
        cv2.circle(overlay, (cx, oy), 7, (0, 0, 255), -1)
        destination = DestinationPointCalculator.calculate(self.config).astype(np.int32)
        cv2.polylines(
            overlay, [destination[[0, 1, 3, 2]]], True, (0, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            overlay,
            "vehicle origin",
            (max(0, cx + 8), max(20, oy - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        return overlay

    def _refresh_static_previews(self) -> None:
        """仅在静态状态变化时重新编码标定图、鸟瞰图和可选 mask。"""
        with self.lock:
            image = self._display_image()
            matrix = None if self.matrix is None else self.matrix.copy()
            if image is None:
                return
            source_message = self._display_image_message()
            calibration_frame = image.copy()
            annotated = self._annotate_calibration_frame(image)
            selected_masks = self._display_masks()
            selected_messages = self._display_mask_messages()
            masks = {
                name: None if value is None else value.copy()
                for name, value in selected_masks.items()
            }
            mask_messages = selected_messages.copy()
        self.frames.update(
            "calibration_frame",
            calibration_frame,
            self.config.web_gui_jpeg_quality,
        )
        self.frames.update(
            "calibration_annotated",
            annotated,
            self.config.web_gui_jpeg_quality,
        )
        if matrix is None:
            blank = np.zeros((480, 600, 3), dtype=np.uint8)
            cv2.putText(
                blank,
                "Select TL TR BL BR",
                (90, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            self.frames.update(
                "calibration_birdview",
                blank,
                self.config.web_gui_jpeg_quality,
            )
            return
        size = (self.config.output_width_px, self.config.output_height_px)
        pure_birdview = IpmTransformer.color(image, matrix, size)
        web_birdview = self._annotate_birdview(pure_birdview)
        self.frames.update(
            "calibration_birdview",
            web_birdview,
            self.config.web_gui_jpeg_quality,
        )
        if self.publishers:
            # ROS raw_birdview 是纯透视结果，不混入 Web 调试标注。
            self.publishers["raw"].publish(
                ImageCodec.to_message(pure_birdview, "bgr8", source_message)
            )
        for name, mask in masks.items():
            if mask is None:
                continue
            warped = IpmTransformer.mask(mask, matrix, size)
            self.frames.update(f"{name}_birdview", warped, self.config.web_gui_jpeg_quality)
            if name == "boundary":
                self.frames.update("boundary_raw", mask, self.config.web_gui_jpeg_quality)
            if self.publishers:
                self.publishers[name].publish(
                    ImageCodec.to_message(warped, "mono8", mask_messages[name])
                )

    def _status_document(self) -> Dict[str, Any]:
        with self.lock:
            destination = DestinationPointCalculator.calculate(self.config)
            last_age = (
                max(0.0, time.monotonic() - self.status.last_frame_time)
                if self.status.last_frame_time
                else None
            )
            document = {
                "interface_version": INTERFACE_VERSION,
                "calibration_valid": self.status.calibration_valid,
                "calibration_reason": self.status.calibration_reason,
                "profile_name": self.config.profile_name,
                "config_path": str(self.config_manager.paths(self.config)[0]),
                "image_topic": self.config.image_topic,
                "image_received": self.status.image_received,
                "image_width": self.status.image_width,
                "image_height": self.status.image_height,
                "input_encoding": self.status.input_encoding,
                "last_frame_age_sec": last_age,
                "output_width": self.config.output_width_px,
                "output_height": self.config.output_height_px,
                "source_points": copy.deepcopy(self.points.points),
                "destination_points": destination.tolist(),
                "calibration_board_width_m": self.config.calibration_board_width_m,
                "calibration_board_length_m": self.config.calibration_board_length_m,
                "calibration_grid_size_m": self.config.calibration_grid_size_m,
                "grid_columns": 0,
                "grid_rows": 0,
                "internal_grid_used": False,
                "corner_marks_only": True,
                "meter_per_pixel": self.config.meter_per_pixel,
                "vehicle_center_x_px": self.config.vehicle_center_x_px,
                "vehicle_origin_y_px": self.config.vehicle_origin_y_px,
                "vehicle_reference_point_name": self.config.vehicle_reference_point_name,
                "board_near_edge_distance_m": self.config.board_near_edge_distance_m,
                "board_center_lateral_offset_m": self.config.board_center_lateral_offset_m,
                "expected_lane_width_m": self.config.expected_lane_width_m,
                "single_boundary_center_offset_m": self.config.single_boundary_center_offset_m,
                "expected_board_width_px": self.config.calibration_board_width_m / self.config.meter_per_pixel,
                "expected_board_length_px": self.config.calibration_board_length_m / self.config.meter_per_pixel,
                "expected_reference_size_px": self.config.calibration_grid_size_m / self.config.meter_per_pixel,
                "homography_determinant": self.validation.homography_determinant,
                "source_quad_area_px": self.validation.source_quad_area_px,
                "boundary_mask_received": self.status.boundary_mask_received,
                "yellow_mask_received": self.status.yellow_mask_received,
                "green_mask_received": self.status.green_mask_received,
                "calibration_frame_captured": self.calibration_frame_captured,
                "calibration_frame_sequence": self.calibration_frame_sequence,
                "calibration_frame_captured_at": self.calibration_frame_captured_at,
                "homography_valid": self.matrix is not None
                and self.inverse_matrix is not None,
                "selected_point_count": len(self.points.points),
                "next_point_name": self.points.next_name(),
                "fps": self.status.fps,
                "processing_time_ms": self.status.processing_time_ms,
                "config_loaded": self.status.config_loaded,
                "config_warning": self.status.config_warning,
                "measurement": self.measurement,
                "last_error": self.status.last_error,
            }
            return document

    def _publish_status(self) -> None:
        if not self.publishers:
            return
        message = String()
        message.data = json.dumps(self._status_document(), ensure_ascii=False)
        self.publishers["status"].publish(message)

    def _config_document(self) -> Dict[str, Any]:
        return asdict(self.config)

    def _apply_config_values(self, values: Dict[str, Any]) -> None:
        candidate = copy.deepcopy(self.config)
        for name, value in values.items():
            if name not in self.WEB_EDITABLE_FIELDS:
                raise CalibrationError(f"不允许通过 Web 修改参数: {name}")
            current = getattr(candidate, name)
            if isinstance(current, str):
                converted: Any = str(value).strip()
            elif isinstance(current, int) and not isinstance(current, bool):
                converted = int(float(value))
            else:
                converted = float(value)
            setattr(candidate, name, converted)
        candidate.validate()
        DestinationPointCalculator.validate(DestinationPointCalculator.calculate(candidate), candidate)
        self.config = candidate
        if (
            self.calibration_frame_captured
            and len(self.points.points) == 4
        ):
            self._recompute()
        elif self.calibration_frame_captured:
            self._refresh_static_previews()
        else:
            self._set_validation_error("参数已修改，请采集静态标定帧")

    def _capture_calibration_frame(self) -> None:
        """原子采集原图、可选 mask 与消息头，并重置旧标定。"""
        if self.latest_image is None:
            raise CalibrationError("尚未收到相机图像，无法采集标定帧")
        self.calibration_frame_bgr = self.latest_image.copy()
        self.calibration_frame_header = (
            copy.deepcopy(self.latest_image_message.header)
            if self.latest_image_message is not None
            else None
        )
        self.calibration_masks = {
            name: None if value is None else value.copy()
            for name, value in self.masks.items()
        }
        self.calibration_mask_messages = {
            name: self._header_snapshot(message)
            for name, message in self.mask_messages.items()
        }
        self.calibration_frame_captured = True
        self.calibration_frame_sequence += 1
        self.calibration_frame_captured_at = datetime.now(
            timezone.utc
        ).isoformat()
        self.points.clear()
        self.measurement = None
        self._set_validation_error("静态标定帧已采集，请依次点击 TL、TR、BL、BR")
        self.status.selected_point_count = 0
        self.status.next_point_name = self.points.next_name()
        self._refresh_static_previews()

    def _add_point(self, x: float, y: float) -> None:
        if not self.calibration_frame_captured or self.calibration_frame_bgr is None:
            raise CalibrationError("请先采集静态标定帧")
        if len(self.points.points) >= 4:
            raise CalibrationError("四个点已经完整，请先撤销或清空")
        self.points.set_point(
            len(self.points.points),
            x,
            y,
            self.calibration_frame_bgr.shape[1],
            self.calibration_frame_bgr.shape[0],
        )
        if len(self.points.points) == 4:
            try:
                self._recompute()
            except CalibrationError:
                self._refresh_static_previews()
                raise
        else:
            self._set_validation_error("四点尚未完整")
            self._refresh_static_previews()

    def _set_points_once(self, values: Any) -> None:
        """一次性校验并应用四个手工坐标，失败时不破坏原状态。"""
        if not self.calibration_frame_captured or self.calibration_frame_bgr is None:
            raise CalibrationError("请先采集静态标定帧")
        if not isinstance(values, list) or len(values) != 4:
            raise CalibrationError("手工坐标必须正好包含 TL、TR、BL、BR 四点")
        candidate_points = CalibrationPointManager()
        for index, value in enumerate(values):
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise CalibrationError(f"{POINT_NAMES[index]} 坐标格式错误")
            candidate_points.set_point(
                index,
                float(value[0]),
                float(value[1]),
                self.calibration_frame_bgr.shape[1],
                self.calibration_frame_bgr.shape[0],
            )
        destination = DestinationPointCalculator.calculate(self.config)
        matrix, inverse, validation = HomographyValidator.compute(
            candidate_points.points,
            destination,
            self.config,
            (
                self.calibration_frame_bgr.shape[1],
                self.calibration_frame_bgr.shape[0],
            ),
        )
        self.points = candidate_points
        self.matrix, self.inverse_matrix, self.validation = (
            matrix,
            inverse,
            validation,
        )
        self.status.calibration_valid = True
        self.status.calibration_reason = validation.reason
        self._refresh_static_previews()

    def _save_config(self) -> Tuple[Path, Path]:
        self._recompute()
        assert self.matrix is not None and self.inverse_matrix is not None
        image = self._display_image()
        assert image is not None
        document = self.config_manager.make_document(
            self.config,
            self.points.points,
            DestinationPointCalculator.calculate(self.config),
            self.matrix,
            self.inverse_matrix,
            self.validation,
            (image.shape[1], image.shape[0]),
        )
        return self.config_manager.save(document, self.config)

    def _load_config(self, missing_ok: bool = False) -> None:
        try:
            path, document = self.config_manager.load(self.config)
        except FileNotFoundError:
            if missing_ok:
                return
            raise CalibrationError("配置文件不存在")
        board = document["calibration_board"]
        output = document["output"]
        roads = document["road_geometry_defaults"]
        candidate = copy.deepcopy(self.config)
        candidate.profile_name = str(document["profile_name"])
        candidate.calibration_board_width_m = float(board["width_m"])
        candidate.calibration_board_length_m = float(board["length_m"])
        candidate.calibration_grid_size_m = float(board.get("grid_size_m", 0.10))
        candidate.board_near_edge_distance_m = float(board["near_edge_distance_m"])
        candidate.board_center_lateral_offset_m = float(board["center_lateral_offset_m"])
        candidate.output_width_px = int(output["width_px"])
        candidate.output_height_px = int(output["height_px"])
        candidate.meter_per_pixel = float(output["meter_per_pixel"])
        candidate.vehicle_center_x_px = float(output["vehicle_center_x_px"])
        candidate.vehicle_origin_y_px = float(output["vehicle_origin_y_px"])
        convention = document.get("coordinate_convention", {})
        candidate.vehicle_reference_point_name = str(
            convention.get("vehicle_reference_point_name", "base_link")
        )
        candidate.expected_lane_width_m = float(roads["expected_lane_width_m"])
        candidate.single_boundary_center_offset_m = float(roads["single_boundary_center_offset_m"])
        candidate.validate()
        source_dict = document["source_points"]
        source = [source_dict[key] for key in POINT_KEYS]
        self.config = candidate
        self.points.load(source)
        image_size = (int(document["image_width"]), int(document["image_height"]))
        matrix, inverse, validation = HomographyValidator.compute(
            source, DestinationPointCalculator.calculate(candidate), candidate, image_size
        )
        saved = np.asarray(document["homography_matrix"], dtype=np.float64)
        if saved.shape != (3, 3) or not np.all(np.isfinite(saved)):
            raise CalibrationError("配置中的 Homography 矩阵非法")
        saved_scale = saved[2, 2] if abs(saved[2, 2]) > 1.0e-12 else 1.0
        matrix_scale = matrix[2, 2] if abs(matrix[2, 2]) > 1.0e-12 else 1.0
        difference = float(
            np.max(np.abs(saved / saved_scale - matrix / matrix_scale))
        )
        self.matrix, self.inverse_matrix, self.validation = matrix, inverse, validation
        self.status.calibration_valid = True
        self.status.calibration_reason = "valid"
        self.status.config_loaded = True
        self.status.config_warning = (
            f"保存矩阵与重算矩阵最大差异 {difference:.3e}，已使用重算结果"
            if difference > 1.0e-5
            else ""
        )
        self.get_logger().info(f"已加载 IPM 配置: {path}")
        if self.calibration_frame_captured:
            self._refresh_static_previews()

    def _handle_api(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            if path == "/api/capture_calibration_frame":
                self._capture_calibration_frame()
            elif path == "/api/undo_point":
                if not self.calibration_frame_captured:
                    raise CalibrationError("请先采集静态标定帧")
                self.points.undo()
                self._set_validation_error("四点已修改，请重新计算")
                self._refresh_static_previews()
            elif path == "/api/clear_points":
                if not self.calibration_frame_captured:
                    raise CalibrationError("请先采集静态标定帧")
                self.points.clear()
                self._set_validation_error("等待选择四个角点")
                self._refresh_static_previews()
            elif path == "/api/add_point":
                self._add_point(
                    float(payload["x"]),
                    float(payload["y"]),
                )
            elif path == "/api/set_points":
                self._set_points_once(payload.get("points"))
            elif path == "/api/set":
                self._apply_config_values(payload)
            elif path in ("/api/recompute", "/api/validate"):
                self._recompute()
            elif path == "/api/save":
                json_path, yaml_path = self._save_config()
                return {
                    "ok": True,
                    "success": True,
                    "message": "配置保存成功",
                    "json_path": str(json_path),
                    "yaml_path": str(yaml_path),
                }
            elif path == "/api/load":
                self._load_config()
            elif path == "/api/reset_defaults":
                self.config = copy.deepcopy(self.defaults)
                self.points.clear()
                self._set_validation_error("已恢复默认参数，请重新选择四点")
                if self.calibration_frame_captured:
                    self._refresh_static_previews()
            elif path == "/api/measure_point":
                x1, y1 = float(payload["x1"]), float(payload["y1"])
                x2, y2 = float(payload["x2"]), float(payload["y2"])
                for x, y in ((x1, y1), (x2, y2)):
                    if not (0 <= x < self.config.output_width_px and 0 <= y < self.config.output_height_px):
                        raise CalibrationError("测距点超出鸟瞰图范围")
                distance = math.hypot(x2 - x1, y2 - y1) * self.config.meter_per_pixel
                pixel_distance = math.hypot(x2 - x1, y2 - y1)
                lateral_distance = abs(x2 - x1) * self.config.meter_per_pixel
                forward_distance = abs(y2 - y1) * self.config.meter_per_pixel
                first_forward = (
                    self.config.vehicle_origin_y_px - y1
                ) * self.config.meter_per_pixel
                first_left = (
                    self.config.vehicle_center_x_px - x1
                ) * self.config.meter_per_pixel
                second_forward = (
                    self.config.vehicle_origin_y_px - y2
                ) * self.config.meter_per_pixel
                second_left = (
                    self.config.vehicle_center_x_px - x2
                ) * self.config.meter_per_pixel
                self.measurement = {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "pixel_distance": pixel_distance,
                    "distance_m": distance,
                    "lateral_distance_m": lateral_distance,
                    "forward_distance_m": forward_distance,
                    "point1_vehicle": {
                        "forward_m": first_forward,
                        "left_m": first_left,
                    },
                    "point2_vehicle": {
                        "forward_m": second_forward,
                        "left_m": second_left,
                    },
                }
                if self.calibration_frame_captured:
                    self._refresh_static_previews()
                return {
                    "ok": True,
                    "success": True,
                    "message": "测距完成",
                    **self.measurement,
                }
            elif path == "/api/clear_measurement":
                self.measurement = None
                if self.calibration_frame_captured:
                    self._refresh_static_previews()
            else:
                raise CalibrationError(f"未知 API: {path}")
            self.status.selected_point_count = len(self.points.points)
            self.status.next_point_name = self.points.next_name()
            return {"ok": True, "success": True, "message": "操作成功"}

    def _start_web(self) -> None:
        node = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: Any) -> None:
                return

            def _json(self, status: int, data: Dict[str, Any]) -> None:
                body = json.dumps(data, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
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
                    return
                if path == "/api/status":
                    self._json(200, node._status_document())
                    return
                if path == "/api/config":
                    self._json(200, node._config_document())
                    return
                static_images = {
                    "/image/calibration_frame.jpg": "calibration_frame",
                    "/image/calibration_annotated.jpg": "calibration_annotated",
                    "/image/calibration_birdview.jpg": "calibration_birdview",
                }
                if path in static_images:
                    self._image(static_images[path])
                    return
                if path.startswith("/stream/"):
                    stream = path.removeprefix("/stream/")
                    aliases = {
                        "raw": "live_raw",
                        "boundary_raw": "boundary_raw",
                        "boundary_birdview": "boundary_birdview",
                        "yellow_birdview": "yellow_birdview",
                        "green_birdview": "green_birdview",
                    }
                    name = aliases.get(stream, stream)
                    self._stream(name)
                    return
                self.send_error(404)

            def _image(self, name: str) -> None:
                frame, _sequence = node.frames.get(name)
                if frame is None:
                    self.send_error(404, "image not available")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(frame)))
                self.send_header(
                    "Cache-Control", "no-store, no-cache, must-revalidate"
                )
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                self.wfile.write(frame)

            def _stream(self, name: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last_sequence = -1
                try:
                    while True:
                        frame, sequence = node.frames.get(name)
                        if frame is not None and sequence != last_sequence:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                            self.wfile.write(frame + b"\r\n")
                            self.wfile.flush()
                            last_sequence = sequence
                        time.sleep(max(0.05, 1.0 / node.config.web_gui_max_fps))
                except (BrokenPipeError, ConnectionResetError):
                    return

            def do_POST(self) -> None:
                path = urllib.parse.urlparse(self.path).path
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length > 65536:
                        raise CalibrationError("请求体过大")
                    raw = self.rfile.read(length) if length else b"{}"
                    payload = json.loads(raw.decode("utf-8"))
                    if not isinstance(payload, dict):
                        raise CalibrationError("请求必须是 JSON 对象")
                    result = node._handle_api(path, payload)
                    self._json(200, result)
                except (CalibrationError, KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
                    node.status.last_error = str(exc)
                    self._json(400, {"ok": False, "error": str(exc)})
                except Exception as exc:
                    node.status.last_error = f"内部错误: {exc}"
                    self._json(500, {"ok": False, "error": "内部错误，请查看节点日志"})

        self.web_server = ThreadingHTTPServer(
            (self.config.web_gui_host, self.config.web_gui_port), Handler
        )
        self.web_thread = threading.Thread(
            target=self.web_server.serve_forever, name="ipm-web", daemon=True
        )
        self.web_thread.start()

    def destroy_node(self) -> bool:
        if self.web_server is not None:
            self.web_server.shutdown()
            self.web_server.server_close()
        return super().destroy_node()


def main(args: Optional[List[str]] = None) -> None:
    """启动独立第三阶段 ROS 节点。"""
    rclpy.init(args=args)
    node: Optional[IpmCalibrationNode] = None
    try:
        node = IpmCalibrationNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if node is not None:
            node.get_logger().error(f"IPM 标定工具异常: {exc}")
        else:
            print(f"IPM 标定工具启动失败: {exc}")
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
