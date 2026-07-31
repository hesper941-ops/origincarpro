#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地静态图 IPM 逆透视标定工具。

用途：
1. 读取从 RDK X5 导出的单张相机图片；
2. 在本地 OpenCV 窗口中依次点击已知地面矩形的四个角；
3. 根据矩形真实尺寸、米/像素比例和车辆参考原点计算目标点；
4. 生成鸟瞰图预览；
5. 保存供 4_lane_geometry_planner.py 使用的 gxb_ipm_v1 JSON 配置。

本工具不依赖 ROS，不包含道路中心线、路径规划或运动控制。

点击顺序：
P1 TL：远端左角
P2 TR：远端右角
P3 BL：近端左角
P4 BR：近端右角

快捷键：
左键：按顺序添加角点
右键 / U：撤销最后一个点
C：清空全部点
R：重新计算鸟瞰图
S：保存 JSON 与预览图片
P：在终端打印当前点坐标
H：显示帮助
Q / ESC：退出

依赖：
pip install opencv-python numpy
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]

SOURCE_POINT_NAMES = (
    "P1 TL 远端左角",
    "P2 TR 远端右角",
    "P3 BL 近端左角",
    "P4 BR 近端右角",
)
SOURCE_POINT_KEYS = ("top_left", "top_right", "bottom_left", "bottom_right")
POINT_COLORS_BGR = ((0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255))
WINDOW_SOURCE = "IPM Offline Calibration - Source"
WINDOW_BIRDVIEW = "IPM Offline Calibration - Birdview"


@dataclass
class CalibrationParameters:
    profile_name: str = "usb_camera"
    board_width_m: float = 0.50
    board_length_m: float = 0.70
    board_near_edge_distance_m: float = 0.20
    board_center_lateral_offset_m: float = 0.0
    output_width_px: int = 600
    output_height_px: int = 800
    meter_per_pixel: float = 0.005
    vehicle_center_x_px: float = 300.0
    vehicle_origin_y_px: float = 799.0
    vehicle_reference_point_name: str = "base_link"
    expected_lane_width_m: float = 0.50
    single_boundary_center_offset_m: float = 0.25
    min_source_quad_area_px: float = 5000.0
    homography_determinant_min_abs: float = 1e-9
    virtual_grid_size_m: float = 0.10


@dataclass
class CalibrationResult:
    source_points: np.ndarray
    destination_points: np.ndarray
    homography_matrix: np.ndarray
    inverse_homography_matrix: np.ndarray
    source_quad_area_px: float
    homography_determinant: float
    birdview_clean: np.ndarray
    birdview_overlay: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="本地静态图 IPM 四点标定工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", type=Path, required=True, help="从 X5 导出的原始相机图片")
    parser.add_argument(
        "--output-config",
        type=Path,
        default=Path("gxb_test/config/usb_camera_ipm.json"),
        help="输出 JSON 配置路径",
    )
    parser.add_argument("--profile-name", default="usb_camera")
    parser.add_argument("--board-width-m", type=float, default=0.50)
    parser.add_argument("--board-length-m", type=float, default=0.70)
    parser.add_argument("--board-near-edge-distance-m", type=float, default=0.20)
    parser.add_argument("--board-center-lateral-offset-m", type=float, default=0.0)
    parser.add_argument("--output-width-px", type=int, default=600)
    parser.add_argument("--output-height-px", type=int, default=800)
    parser.add_argument("--meter-per-pixel", type=float, default=0.005)
    parser.add_argument("--vehicle-center-x-px", type=float, default=300.0)
    parser.add_argument("--vehicle-origin-y-px", type=float, default=799.0)
    parser.add_argument("--vehicle-reference-point-name", default="base_link")
    parser.add_argument("--expected-lane-width-m", type=float, default=0.50)
    parser.add_argument("--single-boundary-center-offset-m", type=float, default=0.25)
    parser.add_argument("--virtual-grid-size-m", type=float, default=0.10)
    parser.add_argument("--load-config", type=Path, default=None)
    return parser.parse_args()


def validate_profile_name(name: str) -> None:
    if not name:
        raise ValueError("profile_name 不能为空")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    if any(ch not in allowed for ch in name):
        raise ValueError("profile_name 仅允许字母、数字、下划线和短横线")


def load_bgr_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"找不到图片：{path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV 无法读取图片：{path}")
    return image


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"图片编码失败：{path}")
    encoded.tofile(str(path))


def polygon_area(points: Sequence[Point]) -> float:
    arr = np.asarray(points, dtype=np.float64)
    x = arr[:, 0]
    y = arr[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(a: Point, b: Point, c: Point, d: Point, eps: float = 1e-9) -> bool:
    o1, o2 = orientation(a, b, c), orientation(a, b, d)
    o3, o4 = orientation(c, d, a), orientation(c, d, b)
    return (o1 * o2 < -eps) and (o3 * o4 < -eps)


def calculate_destination_points(params: CalibrationParameters) -> np.ndarray:
    if params.meter_per_pixel <= 0:
        raise ValueError("meter_per_pixel 必须大于 0")
    if params.board_width_m <= 0 or params.board_length_m <= 0:
        raise ValueError("标定矩形尺寸必须大于 0")
    if params.output_width_px <= 0 or params.output_height_px <= 0:
        raise ValueError("鸟瞰输出尺寸必须大于 0")

    half_width_px = params.board_width_m / params.meter_per_pixel / 2.0
    length_px = params.board_length_m / params.meter_per_pixel
    near_px = params.board_near_edge_distance_m / params.meter_per_pixel
    offset_px = params.board_center_lateral_offset_m / params.meter_per_pixel

    center_x = params.vehicle_center_x_px - offset_px
    bottom_y = params.vehicle_origin_y_px - near_px
    top_y = bottom_y - length_px

    dst = np.array(
        [
            [center_x - half_width_px, top_y],
            [center_x + half_width_px, top_y],
            [center_x - half_width_px, bottom_y],
            [center_x + half_width_px, bottom_y],
        ],
        dtype=np.float32,
    )

    for i, (x, y) in enumerate(dst):
        if not (0 <= float(x) <= params.output_width_px - 1 and 0 <= float(y) <= params.output_height_px - 1):
            raise ValueError(
                f"目的点 {SOURCE_POINT_KEYS[i]} 越界：({x:.1f}, {y:.1f})；"
                "请调整输出尺寸、meter_per_pixel、车辆原点或矩形距离"
            )
    return dst


def validate_source_points(points: Sequence[Point], width: int, height: int, min_area: float) -> float:
    if len(points) != 4:
        raise ValueError("必须选择 4 个角点")
    pts = [(float(x), float(y)) for x, y in points]

    for i, (x, y) in enumerate(pts):
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"{SOURCE_POINT_NAMES[i]} 越界：({x:.1f}, {y:.1f})")

    for i in range(4):
        for j in range(i + 1, 4):
            if math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) < 3.0:
                raise ValueError(f"{SOURCE_POINT_NAMES[i]} 与 {SOURCE_POINT_NAMES[j]} 重合或过近")

    tl, tr, bl, br = pts
    if tl[0] >= tr[0]:
        raise ValueError("TL 必须位于 TR 左侧")
    if bl[0] >= br[0]:
        raise ValueError("BL 必须位于 BR 左侧")
    if (tl[1] + tr[1]) / 2.0 >= (bl[1] + br[1]) / 2.0:
        raise ValueError("远端 TL/TR 必须整体位于近端 BL/BR 上方")
    if segments_intersect(tr, br, bl, tl):
        raise ValueError("左右边发生自交，请检查点位顺序")

    area = polygon_area([tl, tr, br, bl])
    if area < min_area:
        raise ValueError(f"源四边形面积过小：{area:.1f} px²，要求至少 {min_area:.1f} px²")
    return area


def normalize_homography(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Homography 非法")
    if abs(matrix[2, 2]) > 1e-12:
        matrix = matrix / matrix[2, 2]
    return matrix


def draw_text(image: np.ndarray, text: str, origin: Tuple[int, int], color: Tuple[int, int, int], scale: float = 0.55) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def annotate_source(source: np.ndarray, points: Sequence[Point], status: str) -> np.ndarray:
    canvas = source.copy()
    for i, point in enumerate(points):
        x, y = int(round(point[0])), int(round(point[1]))
        color = POINT_COLORS_BGR[i]
        cv2.circle(canvas, (x, y), 7, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 5, color, -1, cv2.LINE_AA)
        draw_text(canvas, SOURCE_POINT_NAMES[i], (x + 10, max(45, y - 10)), color)

    if len(points) >= 2:
        for i in range(len(points) - 1):
            p1 = tuple(int(round(v)) for v in points[i])
            p2 = tuple(int(round(v)) for v in points[i + 1])
            cv2.line(canvas, p1, p2, (255, 255, 0), 2, cv2.LINE_AA)
    if len(points) == 4:
        tl, tr, bl, br = points
        poly = np.asarray([tl, tr, br, bl], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [poly], True, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (20, 20, 20), -1)
    draw_text(canvas, status, (8, 24), (255, 255, 255))
    return canvas


def draw_birdview_overlay(clean: np.ndarray, dst: np.ndarray, params: CalibrationParameters) -> np.ndarray:
    overlay = clean.copy()
    h, w = overlay.shape[:2]

    grid_px = params.virtual_grid_size_m / params.meter_per_pixel if params.virtual_grid_size_m > 0 else 0
    if grid_px >= 5:
        step = max(1, int(round(grid_px)))
        for x in range(int(round(params.vehicle_center_x_px)) % step, w, step):
            cv2.line(overlay, (x, 0), (x, h - 1), (75, 75, 75), 1, cv2.LINE_AA)
        for y in range(int(round(params.vehicle_origin_y_px)), -1, -step):
            cv2.line(overlay, (0, y), (w - 1, y), (75, 75, 75), 1, cv2.LINE_AA)

    cx = int(round(params.vehicle_center_x_px))
    oy = int(round(params.vehicle_origin_y_px))
    cv2.line(overlay, (cx, 0), (cx, h - 1), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(overlay, (cx, oy), 7, (0, 0, 255), -1, cv2.LINE_AA)
    draw_text(overlay, f"origin: {params.vehicle_reference_point_name}", (min(w - 220, cx + 10), max(45, oy - 10)), (0, 0, 255))

    tl, tr, bl, br = dst
    poly = np.asarray([tl, tr, br, bl], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(overlay, [poly], True, (255, 0, 255), 2, cv2.LINE_AA)

    cv2.rectangle(overlay, (0, 0), (w, 34), (20, 20, 20), -1)
    draw_text(
        overlay,
        f"target={params.board_width_m:.2f}m x {params.board_length_m:.2f}m | {params.meter_per_pixel:.4f} m/px",
        (8, 24),
        (255, 255, 255),
    )
    return overlay


def calculate_result(source: np.ndarray, points: Sequence[Point], params: CalibrationParameters) -> CalibrationResult:
    h, w = source.shape[:2]
    area = validate_source_points(points, w, h, params.min_source_quad_area_px)
    src = np.asarray(points, dtype=np.float32)
    dst = calculate_destination_points(params)

    matrix = normalize_homography(cv2.getPerspectiveTransform(src, dst))
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < params.homography_determinant_min_abs:
        raise ValueError(f"Homography 行列式过小：{determinant:.3e}")
    try:
        inverse = normalize_homography(np.linalg.inv(matrix))
    except np.linalg.LinAlgError as exc:
        raise ValueError("Homography 无法求逆") from exc

    clean = cv2.warpPerspective(
        source,
        matrix,
        (params.output_width_px, params.output_height_px),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    overlay = draw_birdview_overlay(clean, dst, params)
    return CalibrationResult(src, dst, matrix, inverse, area, determinant, clean, overlay)


def matrix_to_list(matrix: np.ndarray) -> List[List[float]]:
    return [[float(v) for v in row] for row in np.asarray(matrix, dtype=np.float64)]


def points_to_mapping(points: np.ndarray) -> Dict[str, List[float]]:
    return {key: [float(points[i, 0]), float(points[i, 1])] for i, key in enumerate(SOURCE_POINT_KEYS)}


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def create_config_payload(image_path: Path, source: np.ndarray, params: CalibrationParameters, result: CalibrationResult) -> Dict[str, Any]:
    h, w = source.shape[:2]
    return {
        "version": "gxb_ipm_v1",
        "profile_name": params.profile_name,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "calibration_source_image": str(image_path.resolve()),
        "image_topic": "/image_out",
        "image_width": int(w),
        "image_height": int(h),
        "coordinate_convention": {
            "birdview_pixel_x": "right_positive",
            "birdview_pixel_y": "down_positive",
            "vehicle_forward": "up_in_birdview",
            "vehicle_left": "left_in_birdview",
            "vehicle_origin": "configured_reference_point",
            "vehicle_reference_point_name": params.vehicle_reference_point_name,
            "forward_m_formula": "(vehicle_origin_y_px - birdview_y_px) * meter_per_pixel",
            "left_m_formula": "(vehicle_center_x_px - birdview_x_px) * meter_per_pixel",
        },
        "calibration_board": {
            "width_m": params.board_width_m,
            "length_m": params.board_length_m,
            "has_internal_grid": False,
            "grid_size_m": params.virtual_grid_size_m,
            "grid_columns": 0,
            "grid_rows": 0,
            "near_edge_distance_m": params.board_near_edge_distance_m,
            "center_lateral_offset_m": params.board_center_lateral_offset_m,
            "note": "现场为已知尺寸矩形区域，内部没有实体网格；grid_size_m 仅用于鸟瞰虚拟网格显示。",
        },
        "source_points": points_to_mapping(result.source_points),
        "destination_points": points_to_mapping(result.destination_points),
        "output": {
            "width_px": params.output_width_px,
            "height_px": params.output_height_px,
            "meter_per_pixel": params.meter_per_pixel,
            "vehicle_center_x_px": params.vehicle_center_x_px,
            "vehicle_origin_y_px": params.vehicle_origin_y_px,
        },
        "road_geometry_defaults": {
            "expected_lane_width_m": params.expected_lane_width_m,
            "single_boundary_center_offset_m": params.single_boundary_center_offset_m,
        },
        "homography_matrix": matrix_to_list(result.homography_matrix),
        "inverse_homography_matrix": matrix_to_list(result.inverse_homography_matrix),
        "validation": {
            "valid": True,
            "source_quad_area_px": result.source_quad_area_px,
            "homography_determinant": result.homography_determinant,
            "expected_board_width_px": params.board_width_m / params.meter_per_pixel,
            "expected_board_length_px": params.board_length_m / params.meter_per_pixel,
            "independent_accuracy_note": "四个源角点被强制映射到目标矩形，目标外框尺寸本身不是独立精度证明；请使用未参与四点计算的已知距离或固定赛道宽度额外验证。",
        },
    }


def load_existing_config(path: Path, params: CalibrationParameters) -> Tuple[CalibrationParameters, List[Point]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != "gxb_ipm_v1":
        raise ValueError("仅支持 version=gxb_ipm_v1")

    board = data.get("calibration_board", {})
    output = data.get("output", {})
    road = data.get("road_geometry_defaults", {})
    convention = data.get("coordinate_convention", {})

    params.profile_name = str(data.get("profile_name", params.profile_name))
    params.board_width_m = float(board.get("width_m", params.board_width_m))
    params.board_length_m = float(board.get("length_m", params.board_length_m))
    params.board_near_edge_distance_m = float(board.get("near_edge_distance_m", params.board_near_edge_distance_m))
    params.board_center_lateral_offset_m = float(board.get("center_lateral_offset_m", params.board_center_lateral_offset_m))
    params.virtual_grid_size_m = float(board.get("grid_size_m", params.virtual_grid_size_m))
    params.output_width_px = int(output.get("width_px", params.output_width_px))
    params.output_height_px = int(output.get("height_px", params.output_height_px))
    params.meter_per_pixel = float(output.get("meter_per_pixel", params.meter_per_pixel))
    params.vehicle_center_x_px = float(output.get("vehicle_center_x_px", params.vehicle_center_x_px))
    params.vehicle_origin_y_px = float(output.get("vehicle_origin_y_px", params.vehicle_origin_y_px))
    params.expected_lane_width_m = float(road.get("expected_lane_width_m", params.expected_lane_width_m))
    params.single_boundary_center_offset_m = float(road.get("single_boundary_center_offset_m", params.single_boundary_center_offset_m))
    params.vehicle_reference_point_name = str(convention.get("vehicle_reference_point_name", params.vehicle_reference_point_name))

    mapping = data.get("source_points", {})
    points: List[Point] = []
    for key in SOURCE_POINT_KEYS:
        value = mapping.get(key)
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"配置缺少 source_points.{key}")
        points.append((float(value[0]), float(value[1])))
    return params, points


class OfflineIpmCalibrator:
    def __init__(
        self,
        source: np.ndarray,
        image_path: Path,
        output_config: Path,
        params: CalibrationParameters,
        initial_points: Optional[Sequence[Point]] = None,
    ) -> None:
        self.source = source
        self.image_path = image_path
        self.output_config = output_config
        self.params = params
        self.points: List[Point] = list(initial_points or [])
        self.result: Optional[CalibrationResult] = None
        self.last_error = ""
        if len(self.points) == 4:
            self.recompute()

    def mouse_callback(self, event: int, x: int, y: int, flags: int, userdata: Any) -> None:
        del flags, userdata
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) >= 4:
                print("已经有四个点；请按 U 撤销或 C 清空")
                return
            self.points.append((float(x), float(y)))
            print(f"已添加 {SOURCE_POINT_NAMES[len(self.points) - 1]}：({x}, {y})")
            self.result = None
            self.last_error = ""
            if len(self.points) == 4:
                self.recompute()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.undo()

    def undo(self) -> None:
        if not self.points:
            print("当前没有可撤销的点")
            return
        i = len(self.points) - 1
        point = self.points.pop()
        print(f"已撤销 {SOURCE_POINT_NAMES[i]}：({point[0]:.0f}, {point[1]:.0f})")
        self.result = None
        self.last_error = ""

    def clear(self) -> None:
        self.points.clear()
        self.result = None
        self.last_error = ""
        print("已清空全部点")

    def recompute(self) -> None:
        try:
            self.result = calculate_result(self.source, self.points, self.params)
            self.last_error = ""
            print("Homography 计算成功；按 S 保存")
        except Exception as exc:
            self.result = None
            self.last_error = str(exc)
            print(f"计算失败：{exc}")

    def print_points(self) -> None:
        print("\n当前四点：")
        for i, point in enumerate(self.points):
            print(f"  {SOURCE_POINT_NAMES[i]}: ({point[0]:.1f}, {point[1]:.1f})")

    def save(self) -> None:
        if self.result is None:
            self.recompute()
        if self.result is None:
            raise RuntimeError(self.last_error or "没有有效标定结果")

        payload = create_config_payload(self.image_path, self.source, self.params, self.result)
        atomic_write_json(self.output_config, payload)

        stem = self.output_config.stem
        parent = self.output_config.parent
        source_preview = parent / f"{stem}_source_annotated.png"
        clean_preview = parent / f"{stem}_birdview_clean.png"
        overlay_preview = parent / f"{stem}_birdview_overlay.png"

        imwrite_unicode(source_preview, annotate_source(self.source, self.points, "saved calibration points"))
        imwrite_unicode(clean_preview, self.result.birdview_clean)
        imwrite_unicode(overlay_preview, self.result.birdview_overlay)

        print("\n保存成功：")
        print(f"  JSON：{self.output_config.resolve()}")
        print(f"  原图标注：{source_preview.resolve()}")
        print(f"  纯鸟瞰图：{clean_preview.resolve()}")
        print(f"  鸟瞰调试图：{overlay_preview.resolve()}")

    def source_view(self) -> np.ndarray:
        if len(self.points) < 4:
            status = f"next: {SOURCE_POINT_NAMES[len(self.points)]}"
        elif self.result is not None:
            status = "valid | S save | U undo | C clear | Q quit"
        else:
            status = f"invalid: {self.last_error}"
        return annotate_source(self.source, self.points, status)

    def empty_birdview(self) -> np.ndarray:
        canvas = np.zeros((self.params.output_height_px, self.params.output_width_px, 3), dtype=np.uint8)
        draw_text(canvas, "Birdview unavailable", (20, 55), (255, 255, 255), 0.7)
        draw_text(canvas, "Select 4 points: TL, TR, BL, BR", (20, 95), (255, 255, 255), 0.6)
        if self.last_error:
            draw_text(canvas, self.last_error[:75], (20, 135), (0, 0, 255), 0.5)
        return canvas

    def run(self) -> None:
        cv2.namedWindow(WINDOW_SOURCE, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(WINDOW_BIRDVIEW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_BIRDVIEW, min(self.params.output_width_px, 700), min(self.params.output_height_px, 850))
        cv2.setMouseCallback(WINDOW_SOURCE, self.mouse_callback)
        print_help()

        while True:
            cv2.imshow(WINDOW_SOURCE, self.source_view())
            cv2.imshow(WINDOW_BIRDVIEW, self.result.birdview_overlay if self.result is not None else self.empty_birdview())
            key = cv2.waitKey(30) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("u"):
                self.undo()
            elif key == ord("c"):
                self.clear()
            elif key == ord("r"):
                self.recompute()
            elif key == ord("s"):
                try:
                    self.save()
                except Exception as exc:
                    self.last_error = str(exc)
                    print(f"保存失败：{exc}", file=sys.stderr)
            elif key == ord("p"):
                self.print_points()
            elif key == ord("h"):
                print_help()

        cv2.destroyAllWindows()


def print_help() -> None:
    print(
        """
================ 本地 IPM 标定操作 ================
左键：依次添加 TL、TR、BL、BR
右键 / U：撤销最后一个点
C：清空四点
R：重新计算
S：保存 JSON 和三张预览图
P：打印当前点坐标
H：再次显示帮助
Q / ESC：退出

注意：
- TL/TR 是离车辆更远的两个角；
- BL/BR 是离车辆更近的两个角；
- board_near_edge_distance_m 应从 vehicle_reference_point_name
  指定的车辆参考原点量到标定矩形近边；
- 目标矩形 100×140 px 是构造结果，不是独立精度证明。
====================================================
"""
    )


def build_parameters(args: argparse.Namespace) -> CalibrationParameters:
    params = CalibrationParameters(
        profile_name=args.profile_name,
        board_width_m=args.board_width_m,
        board_length_m=args.board_length_m,
        board_near_edge_distance_m=args.board_near_edge_distance_m,
        board_center_lateral_offset_m=args.board_center_lateral_offset_m,
        output_width_px=args.output_width_px,
        output_height_px=args.output_height_px,
        meter_per_pixel=args.meter_per_pixel,
        vehicle_center_x_px=args.vehicle_center_x_px,
        vehicle_origin_y_px=args.vehicle_origin_y_px,
        vehicle_reference_point_name=args.vehicle_reference_point_name,
        expected_lane_width_m=args.expected_lane_width_m,
        single_boundary_center_offset_m=args.single_boundary_center_offset_m,
        virtual_grid_size_m=args.virtual_grid_size_m,
    )
    validate_profile_name(params.profile_name)
    return params


def main() -> int:
    args = parse_args()
    try:
        params = build_parameters(args)
        points: List[Point] = []
        if args.load_config is not None:
            params, points = load_existing_config(args.load_config, params)
            validate_profile_name(params.profile_name)
            print(f"已加载配置：{args.load_config}")

        source = load_bgr_image(args.image)
        h, w = source.shape[:2]
        print(f"读取图片：{args.image.resolve()}")
        print(f"分辨率：{w}×{h}")
        print(f"标定矩形：{params.board_width_m:.2f} m × {params.board_length_m:.2f} m")
        print(
            "目标尺寸："
            f"{params.board_width_m / params.meter_per_pixel:.1f} px × "
            f"{params.board_length_m / params.meter_per_pixel:.1f} px"
        )
        print(f"车辆参考点：{params.vehicle_reference_point_name}")
        print(f"近边距离：{params.board_near_edge_distance_m:.3f} m")

        OfflineIpmCalibrator(source, args.image, args.output_config, params, points).run()
        return 0
    except Exception as exc:
        print(f"离线 IPM 标定工具启动失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
