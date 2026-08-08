"""Pure cone bbox validation and ground-position fusion mathematics."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FusionResult:
    x: float
    y: float
    ipm_raw_distance: float
    ipm_center_distance: float
    height_distance: float
    fused_distance: float


def bbox_is_reliable(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    image_width: int,
    image_height: int,
    edge_margin_px: float,
) -> bool:
    """Return whether a bbox is complete enough for absolute localization."""
    values = (xmin, ymin, xmax, ymax, image_width, image_height, edge_margin_px)
    if not all(math.isfinite(float(value)) for value in values):
        return False
    if image_width <= 0 or image_height <= 0 or edge_margin_px < 0.0:
        return False
    if xmin < 0.0 or ymin < 0.0 or xmax > image_width or ymax > image_height:
        return False
    if xmax <= xmin or ymax <= ymin:
        return False
    return not (
        xmin <= edge_margin_px
        or ymin <= edge_margin_px
        or xmax >= image_width - edge_margin_px
        or ymax >= image_height - edge_margin_px
    )


def center_within_ground_fov(
    x: float,
    y: float,
    boundary_slope_y_per_x: float,
    footprint_radius_m: float,
) -> bool:
    """Test a circular footprint against both sloped sides of a wedge FOV."""
    values = (x, y, boundary_slope_y_per_x, footprint_radius_m)
    if not all(math.isfinite(float(value)) for value in values):
        return False
    if x <= 0.0 or boundary_slope_y_per_x <= 0.0 or footprint_radius_m < 0.0:
        return False
    required_line_numerator = footprint_radius_m * math.sqrt(
        1.0 + boundary_slope_y_per_x * boundary_slope_y_per_x
    )
    return (
        boundary_slope_y_per_x * x - abs(y)
        >= required_line_numerator
    )


def ground_measurement_is_valid(
    x: float,
    y: float,
    min_x_m: float,
    max_x_m: float,
    enable_fov_gate: bool,
    boundary_slope_y_per_x: float,
    footprint_radius_m: float,
) -> bool:
    """Apply range checks and the optional, experimental wedge-FOV gate."""
    values = (x, y, min_x_m, max_x_m)
    if not all(math.isfinite(float(value)) for value in values):
        return False
    if min_x_m < 0.0 or max_x_m <= min_x_m:
        return False
    if x <= min_x_m or x > max_x_m:
        return False
    if not enable_fov_gate:
        return True
    return center_within_ground_fov(
        x, y, boundary_slope_y_per_x, footprint_radius_m
    )


def fuse_cone_position(
    point_x: float,
    point_y: float,
    bbox_height_px: float,
    camera_ground_x_m: float,
    camera_ground_y_m: float,
    ipm_center_offset_m: float,
    height_model_a: float,
    height_model_b: float,
    fusion_alpha_ipm: float,
    min_ipm_radius_m: float = 1.0e-6,
) -> FusionResult:
    """Fuse IPM-near-edge and bbox-height range along the IPM ground ray."""
    values = (
        point_x,
        point_y,
        bbox_height_px,
        camera_ground_x_m,
        camera_ground_y_m,
        ipm_center_offset_m,
        height_model_a,
        height_model_b,
        fusion_alpha_ipm,
        min_ipm_radius_m,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("fusion inputs must be finite")
    if bbox_height_px <= 0.0:
        raise ValueError("bbox height must be positive")
    if ipm_center_offset_m < 0.0 or height_model_a <= 0.0:
        raise ValueError("invalid physical or height-model parameter")
    if not 0.0 <= fusion_alpha_ipm <= 1.0:
        raise ValueError("fusion_alpha_ipm must be within [0,1]")
    if min_ipm_radius_m <= 0.0:
        raise ValueError("min_ipm_radius_m must be positive")

    dx = point_x - camera_ground_x_m
    dy = point_y - camera_ground_y_m
    ipm_raw_distance = math.hypot(dx, dy)
    if ipm_raw_distance <= min_ipm_radius_m:
        raise ValueError("IPM point is too close to the camera ground projection")

    unit_x = dx / ipm_raw_distance
    unit_y = dy / ipm_raw_distance
    ipm_center_distance = ipm_raw_distance + ipm_center_offset_m
    height_distance = height_model_a / bbox_height_px + height_model_b
    fused_distance = (
        fusion_alpha_ipm * ipm_center_distance
        + (1.0 - fusion_alpha_ipm) * height_distance
    )
    if fused_distance <= 0.0 or not math.isfinite(fused_distance):
        raise ValueError("fused distance must be positive and finite")

    return FusionResult(
        x=camera_ground_x_m + fused_distance * unit_x,
        y=camera_ground_y_m + fused_distance * unit_y,
        ipm_raw_distance=ipm_raw_distance,
        ipm_center_distance=ipm_center_distance,
        height_distance=height_distance,
        fused_distance=fused_distance,
    )
