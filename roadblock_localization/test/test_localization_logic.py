import math

import pytest
from roadblock_interfaces.msg import RoadblockArray

from roadblock_localization.cone_position_fusion import (
    AdaptiveIPMParameters,
    adaptive_ipm_position,
    bbox_is_reliable,
    center_within_ground_fov,
    compute_adaptive_center_offset,
    cone_position_for_model,
    fuse_cone_position,
    ground_measurement_is_valid,
)
from roadblock_localization.roadblock_ground_localizer import (
    build_output_message,
    detection_to_ground_pixel,
)
from roadblock_localization.roadblock_tracker import RoadblockTracker


def make_tracker(ttl=2.0):
    return RoadblockTracker(association_max_distance_m=0.30, track_ttl_sec=ttl)


def adaptive_parameters():
    return AdaptiveIPMParameters(
        c0_m=0.0473222480,
        cr=-0.0505560946,
        cw_m=-0.3862318973,
        ca_m=0.1577719079,
        cu_m=0.2329041982,
        raw_distance_center_m=1.0,
        width_range_center=100.0,
        width_range_scale=50.0,
        aspect_center=0.78,
        aspect_scale=0.15,
        offset_min_m=-1.0,
        offset_max_m=1.0,
    )


def test_bbox_bottom_center_and_invalid_height():
    assert detection_to_ground_pixel(100, 20, 80, 120) == (140.0, 140.0)
    with pytest.raises(ValueError):
        detection_to_ground_pixel(0, 0, 10, 0)
    with pytest.raises(ValueError, match="bbox height"):
        fuse_cone_position(0.65, 0.0, 0.0, 0.05, 0.0, 0.10, 99.488, 0.22381, 0.70)


def test_fusion_formula_and_fixed_weights():
    result = fuse_cone_position(0.65, 0.0, 100.0, 0.05, 0.0, 0.10, 99.488, 0.22381, 0.70)
    assert result.ipm_raw_distance == pytest.approx(0.60)
    assert result.ipm_center_distance == pytest.approx(0.70)
    assert result.height_distance == pytest.approx(99.488 / 100.0 + 0.22381)
    expected = 0.70 * 0.70 + 0.30 * (99.488 / 100.0 + 0.22381)
    assert result.fused_distance == pytest.approx(expected)
    assert result.x == pytest.approx(0.05 + expected)
    assert result.y == pytest.approx(0.0)


def test_fusion_preserves_left_positive_right_negative_y():
    left = fuse_cone_position(0.65, 0.20, 100.0, 0.05, 0.0, 0.10, 99.488, 0.22381, 0.70)
    right = fuse_cone_position(0.65, -0.20, 100.0, 0.05, 0.0, 0.10, 99.488, 0.22381, 0.70)
    assert left.y > 0.0
    assert right.y < 0.0


def test_adaptive_offset_matches_fixed_linear_feature_formula():
    parameters = adaptive_parameters()
    result = compute_adaptive_center_offset(0.80, 100.0, 125.0, 400.0, 640.0, parameters)
    u_norm = 0.25
    aspect = 0.8
    width_times_raw = 80.0
    expected = (
        parameters.c0_m
        + parameters.cr * (0.80 - 1.0)
        + parameters.cw_m * ((width_times_raw - 100.0) / 50.0)
        + parameters.ca_m * ((aspect - 0.78) / 0.15)
        + parameters.cu_m * u_norm**2
    )
    assert result.offset_m == pytest.approx(expected)
    assert result.u_norm == pytest.approx(u_norm)
    assert result.aspect == pytest.approx(aspect)
    assert result.width_times_raw == pytest.approx(width_times_raw)


def test_adaptive_horizontal_feature_is_symmetric_and_zero_at_image_center():
    parameters = adaptive_parameters()
    left = compute_adaptive_center_offset(0.9, 90.0, 120.0, 240.0, 640.0, parameters)
    right = compute_adaptive_center_offset(0.9, 90.0, 120.0, 400.0, 640.0, parameters)
    center = compute_adaptive_center_offset(0.9, 90.0, 120.0, 320.0, 640.0, parameters)
    assert left.u_norm == pytest.approx(-right.u_norm)
    assert left.offset_m == pytest.approx(right.offset_m)
    assert center.u_norm == 0.0


def test_adaptive_invalid_bbox_height_is_rejected_without_nan():
    with pytest.raises(ValueError, match="bbox dimensions"):
        compute_adaptive_center_offset(0.8, 100.0, 0.0, 320.0, 640.0, adaptive_parameters())


def test_adaptive_mode_ignores_legacy_height_and_fusion_parameters():
    common = (
        "adaptive_ipm",
        0.75,
        0.20,
        90.0,
        120.0,
        320.0,
        640.0,
        0.05,
        0.0,
        adaptive_parameters(),
    )
    first = cone_position_for_model(*common, 0.10, 99.488, 0.22381, 0.70, 0.01)
    second = cone_position_for_model(*common, 0.50, 1.0, 9.0, 0.01, 0.01)
    assert second.final_distance == pytest.approx(first.final_distance)
    assert second.x == pytest.approx(first.x)
    assert second.y == pytest.approx(first.y)


def test_legacy_model_selection_preserves_previous_fusion_result():
    selected = cone_position_for_model(
        "legacy_fusion",
        0.65,
        0.0,
        80.0,
        100.0,
        320.0,
        640.0,
        0.05,
        0.0,
        adaptive_parameters(),
        0.10,
        99.488,
        0.22381,
        0.70,
        0.01,
    )
    direct = fuse_cone_position(0.65, 0.0, 100.0, 0.05, 0.0, 0.10, 99.488, 0.22381, 0.70)
    assert selected == direct


def test_adaptive_changes_only_ipm_ray_radius_and_preserves_camera_origin_geometry():
    result = adaptive_ipm_position(
        0.70, 0.25, 90.0, 120.0, 380.0, 640.0, 0.05, 0.0, adaptive_parameters(), 0.01
    )
    raw_angle = math.atan2(0.25, 0.70 - 0.05)
    final_angle = math.atan2(result.y, result.x - 0.05)
    assert final_angle == pytest.approx(raw_angle)
    assert math.hypot(result.x - 0.05, result.y) == pytest.approx(result.final_distance)


def test_cropped_bbox_is_not_reliable():
    assert bbox_is_reliable(20, 30, 100, 200, 640, 400, 2)
    assert not bbox_is_reliable(0, 30, 100, 200, 640, 400, 2)
    assert not bbox_is_reliable(20, 30, 640, 200, 640, 400, 2)
    assert not bbox_is_reliable(20, 30, 100, 401, 640, 400, 2)
    assert not bbox_is_reliable(20, 30, 100, 30, 640, 400, 2)


def test_wedge_fov_uses_perpendicular_footprint_clearance():
    radius = math.hypot(0.10, 0.10)
    x = 0.47
    lateral_limit = 0.713 * x - radius * math.sqrt(1.0 + 0.713**2)
    assert center_within_ground_fov(x, lateral_limit - 1.0e-6, 0.713, radius)
    assert not center_within_ground_fov(x, lateral_limit + 1.0e-6, 0.713, radius)


def test_fov_gate_switch_behaviour():
    radius = math.hypot(0.10, 0.10)
    assert ground_measurement_is_valid(0.70, 0.60, 0.20, 2.00, False, 0.713, radius)
    assert not ground_measurement_is_valid(0.70, 0.60, 0.20, 2.00, True, 0.713, radius)


def test_current_reliable_measurement_is_published_without_odom():
    tracker = make_tracker()
    items = tracker.associate_current_measurements([(1.0, 0.1)], 0.0)
    message = build_output_message(items, RoadblockArray().header.stamp, "base_link")
    assert [(item.id, item.x, item.y) for item in message.obstacles] == [
        (1, pytest.approx(1.0), pytest.approx(0.1))
    ]


def test_no_current_measurement_publishes_empty_but_keeps_id_memory():
    tracker = make_tracker()
    assert tracker.associate_current_measurements([(1.0, 0.0)], 0.0)[0]["id"] == 1
    assert tracker.associate_current_measurements([], 0.2) == []
    assert tracker.tracks[0].id == 1


def test_cropped_frame_never_outputs_previous_track():
    tracker = make_tracker()
    tracker.associate_current_measurements([(1.0, 0.0)], 0.0)
    assert not bbox_is_reliable(0, 30, 100, 200, 640, 400, 2)
    assert tracker.associate_current_measurements([], 0.1) == []


def test_measurement_reappears_with_same_short_term_id():
    tracker = make_tracker()
    first = tracker.associate_current_measurements([(1.0, 0.0)], 0.0)
    hidden = tracker.associate_current_measurements([], 0.5)
    recovered = tracker.associate_current_measurements([(1.05, 0.02)], 1.0)
    assert first[0]["id"] == 1
    assert hidden == []
    assert recovered[0]["id"] == 1


def test_two_current_measurements_keep_ids_when_distance_order_changes():
    tracker = make_tracker()
    first = tracker.associate_current_measurements([(0.80, 0.50), (1.00, -0.50)], 0.0)
    second = tracker.associate_current_measurements([(1.05, 0.50), (0.75, -0.50)], 0.1)
    assert [item["id"] for item in first] == [1, 2]
    assert [item["id"] for item in second] == [1, 2]
    assert math.hypot(second[1]["x"], second[1]["y"]) < math.hypot(
        second[0]["x"], second[0]["y"]
    )


def test_unreliable_measurement_is_never_passed_to_output():
    tracker = make_tracker()
    valid_measurements = []
    if bbox_is_reliable(0, 30, 100, 200, 640, 400, 2):
        valid_measurements.append((1.0, 0.0))
    assert tracker.associate_current_measurements(valid_measurements, 0.0) == []


def test_expired_memory_gets_new_non_reused_id():
    tracker = make_tracker(ttl=1.0)
    assert tracker.associate_current_measurements([(1.0, 0.0)], 0.0)[0]["id"] == 1
    assert tracker.associate_current_measurements([], 1.01) == []
    assert tracker.associate_current_measurements([(1.0, 0.0)], 1.1)[0]["id"] == 2
