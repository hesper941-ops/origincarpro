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


@dataclass(frozen=True)
class AdaptiveIPMParameters:
    c0_m: float
    cr: float
    cw_m: float
    ca_m: float
    cu_m: float
    raw_distance_center_m: float
    width_range_center: float
    width_range_scale: float
    aspect_center: float
    aspect_scale: float
    offset_min_m: float
    offset_max_m: float


@dataclass(frozen=True)
class AdaptiveOffsetResult:
    offset_m: float
    unclamped_offset_m: float
    u_norm: float
    aspect: float
    width_times_raw: float
    was_clamped: bool


@dataclass(frozen=True)
class AdaptiveIPMResult:
    x: float
    y: float
    ipm_raw_distance: float
    adaptive_center_offset: float
    final_distance: float
    u_norm: float
    aspect: float
    width_times_raw: float
    offset_was_clamped: bool


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


def compute_adaptive_center_offset(
    ipm_raw_distance_m: float,
    bbox_width_px: float,
    bbox_height_px: float,
    bbox_center_u_px: float,
    image_width_px: float,
    parameters: AdaptiveIPMParameters,
) -> AdaptiveOffsetResult:
    """Compute the data-fitted radial cone-centre offset from observable bbox geometry."""
    values = (
        ipm_raw_distance_m,
        bbox_width_px,
        bbox_height_px,
        bbox_center_u_px,
        image_width_px,
        parameters.c0_m,
        parameters.cr,
        parameters.cw_m,
        parameters.ca_m,
        parameters.cu_m,
        parameters.raw_distance_center_m,
        parameters.width_range_center,
        parameters.width_range_scale,
        parameters.aspect_center,
        parameters.aspect_scale,
        parameters.offset_min_m,
        parameters.offset_max_m,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("adaptive IPM inputs and parameters must be finite")
    if ipm_raw_distance_m <= 0.0 or bbox_width_px <= 0.0 or bbox_height_px <= 0.0:
        raise ValueError("adaptive IPM distances and bbox dimensions must be positive")
    if image_width_px <= 0.0:
        raise ValueError("image width must be positive")
    if parameters.width_range_scale <= 0.0 or parameters.aspect_scale <= 0.0:
        raise ValueError("adaptive feature scales must be positive")
    if parameters.offset_max_m < parameters.offset_min_m:
        raise ValueError("adaptive offset limits are reversed")

    u_norm = (bbox_center_u_px - image_width_px / 2.0) / (image_width_px / 2.0)
    aspect = bbox_width_px / bbox_height_px
    width_times_raw = bbox_width_px * ipm_raw_distance_m
    raw_offset = (
        parameters.c0_m
        + parameters.cr * (ipm_raw_distance_m - parameters.raw_distance_center_m)
        + parameters.cw_m
        * (
            (width_times_raw - parameters.width_range_center)
            / parameters.width_range_scale
        )
        + parameters.ca_m
        * ((aspect - parameters.aspect_center) / parameters.aspect_scale)
        + parameters.cu_m * u_norm * u_norm
    )
    if not math.isfinite(raw_offset):
        raise ValueError("adaptive offset is not finite")
    offset = min(max(raw_offset, parameters.offset_min_m), parameters.offset_max_m)
    return AdaptiveOffsetResult(
        offset_m=offset,
        unclamped_offset_m=raw_offset,
        u_norm=u_norm,
        aspect=aspect,
        width_times_raw=width_times_raw,
        was_clamped=offset != raw_offset,
    )


def adaptive_ipm_position(
    point_x: float,
    point_y: float,
    bbox_width_px: float,
    bbox_height_px: float,
    bbox_center_u_px: float,
    image_width_px: float,
    camera_ground_x_m: float,
    camera_ground_y_m: float,
    parameters: AdaptiveIPMParameters,
    min_ipm_radius_m: float = 1.0e-6,
) -> AdaptiveIPMResult:
    """Move only radially along the calibrated IPM ray to the cone base centre."""
    values = (
        point_x,
        point_y,
        camera_ground_x_m,
        camera_ground_y_m,
        min_ipm_radius_m,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("adaptive IPM ground inputs must be finite")
    if min_ipm_radius_m <= 0.0:
        raise ValueError("min_ipm_radius_m must be positive")

    dx = point_x - camera_ground_x_m
    dy = point_y - camera_ground_y_m
    ipm_raw_distance = math.hypot(dx, dy)
    if ipm_raw_distance <= min_ipm_radius_m:
        raise ValueError("IPM point is too close to the camera ground projection")
    offset = compute_adaptive_center_offset(
        ipm_raw_distance,
        bbox_width_px,
        bbox_height_px,
        bbox_center_u_px,
        image_width_px,
        parameters,
    )
    final_distance = ipm_raw_distance + offset.offset_m
    if final_distance <= 0.0 or not math.isfinite(final_distance):
        raise ValueError("adaptive final distance must be positive and finite")
    unit_x = dx / ipm_raw_distance
    unit_y = dy / ipm_raw_distance
    return AdaptiveIPMResult(
        x=camera_ground_x_m + final_distance * unit_x,
        y=camera_ground_y_m + final_distance * unit_y,
        ipm_raw_distance=ipm_raw_distance,
        adaptive_center_offset=offset.offset_m,
        final_distance=final_distance,
        u_norm=offset.u_norm,
        aspect=offset.aspect,
        width_times_raw=offset.width_times_raw,
        offset_was_clamped=offset.was_clamped,
    )


def cone_position_for_model(
    distance_model: str,
    point_x: float,
    point_y: float,
    bbox_width_px: float,
    bbox_height_px: float,
    bbox_center_u_px: float,
    image_width_px: float,
    camera_ground_x_m: float,
    camera_ground_y_m: float,
    adaptive_parameters: AdaptiveIPMParameters,
    legacy_ipm_center_offset_m: float,
    legacy_height_model_a: float,
    legacy_height_model_b: float,
    legacy_fusion_alpha_ipm: float,
    min_ipm_radius_m: float,
):
    """Select the formal adaptive model or the unchanged legacy rollback path."""
    if distance_model == "adaptive_ipm":
        return adaptive_ipm_position(
            point_x,
            point_y,
            bbox_width_px,
            bbox_height_px,
            bbox_center_u_px,
            image_width_px,
            camera_ground_x_m,
            camera_ground_y_m,
            adaptive_parameters,
            min_ipm_radius_m,
        )
    if distance_model == "legacy_fusion":
        return fuse_cone_position(
            point_x,
            point_y,
            bbox_height_px,
            camera_ground_x_m,
            camera_ground_y_m,
            legacy_ipm_center_offset_m,
            legacy_height_model_a,
            legacy_height_model_b,
            legacy_fusion_alpha_ipm,
            min_ipm_radius_m,
        )
    raise ValueError("unknown distance model")
