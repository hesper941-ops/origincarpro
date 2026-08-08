import math

import pytest
from roadblock_interfaces.msg import RoadblockArray

from roadblock_localization.cone_position_fusion import (
    bbox_is_reliable,
    center_within_ground_fov,
    fuse_cone_position,
    ground_measurement_is_valid,
)
from roadblock_localization.roadblock_ground_localizer import (
    build_output_message,
    detection_to_ground_pixel,
    update_tracker_if_odom_ready,
)
from roadblock_localization.roadblock_tracker import OdomPose, RoadblockTracker


def make_tracker(ttl=2.0):
    return RoadblockTracker(
        association_max_distance_m=0.30,
        track_ttl_sec=ttl,
        track_min_x_m=-0.30,
        track_max_distance_m=3.0,
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
    assert center_within_ground_fov(x, -(lateral_limit - 1.0e-6), 0.713, radius)


def test_disabled_fov_gate_does_not_reject_outside_wedge():
    radius = math.hypot(0.10, 0.10)
    assert ground_measurement_is_valid(0.70, 0.60, 0.20, 2.00, False, 0.713, radius)


def test_enabled_fov_gate_rejects_outside_wedge():
    radius = math.hypot(0.10, 0.10)
    assert not ground_measurement_is_valid(0.70, 0.60, 0.20, 2.00, True, 0.713, radius)


def test_stable_ids_do_not_follow_distance_order():
    tracker = make_tracker()
    tracker.update([(1.0, 0.0), (1.2, 0.5)], OdomPose(0.0, 0.0, 0.0), 0.0)
    initial = tracker.snapshot(OdomPose(0.0, 0.0, 0.0), 0.1)
    moved_robot = tracker.snapshot(OdomPose(0.8, 0.5, 0.0), 0.2)
    assert [item["id"] for item in initial] == [1, 2]
    assert math.hypot(moved_robot[1]["x"], moved_robot[1]["y"]) < math.hypot(
        moved_robot[0]["x"], moved_robot[0]["y"]
    )
    assert [item["id"] for item in moved_robot] == [1, 2]


def test_odom_translation_propagates_static_track():
    tracker = make_tracker()
    tracker.update([(1.5, 0.0)], OdomPose(0.0, 0.0, 0.0), 0.0)
    item = tracker.snapshot(OdomPose(0.5, 0.0, 0.0), 0.2)[0]
    assert item["x"] == pytest.approx(1.0)
    assert item["y"] == pytest.approx(0.0)


def test_odom_rotation_sign():
    tracker = make_tracker()
    tracker.update([(1.0, 0.0)], OdomPose(0.0, 0.0, 0.0), 0.0)
    item = tracker.snapshot(OdomPose(0.0, 0.0, math.pi / 2.0), 0.2)[0]
    assert item["x"] == pytest.approx(0.0, abs=1.0e-12)
    assert item["y"] == pytest.approx(-1.0)


def test_unreliable_frame_cannot_overwrite_reliable_odom_position():
    tracker = make_tracker()
    tracker.update([(1.0, 0.2)], OdomPose(0.0, 0.0, 0.0), 0.0)
    original = tracker.tracks[0]
    original_position = (original.odom_x, original.odom_y, original.last_valid_time)
    tracker.update([], OdomPose(0.2, 0.0, 0.0), 0.5)
    current = tracker.tracks[0]
    assert (current.odom_x, current.odom_y, current.last_valid_time) == original_position


def test_track_ttl_expires_and_ids_are_not_reused():
    tracker = make_tracker(ttl=1.0)
    tracker.update([(1.0, 0.0)], OdomPose(0.0, 0.0, 0.0), 0.0)
    assert tracker.snapshot(OdomPose(0.0, 0.0, 0.0), 1.01) == []
    tracker.update([(1.1, 0.0)], OdomPose(0.0, 0.0, 0.0), 1.1)
    assert tracker.snapshot(OdomPose(0.0, 0.0, 0.0), 1.1)[0]["id"] == 2


def test_empty_track_pool_builds_empty_array():
    tracker = make_tracker()
    items = tracker.snapshot(OdomPose(0.0, 0.0, 0.0), 0.0)
    message = build_output_message(items, RoadblockArray().header.stamp, "base_link")
    assert message.header.frame_id == "base_link"
    assert list(message.obstacles) == []


def test_no_odom_first_frame_cannot_create_track():
    tracker = make_tracker()
    updated = update_tracker_if_odom_ready(tracker, [(1.0, 0.0)], None, 0.0)
    assert not updated
    assert tracker.tracks == ()

    updated = update_tracker_if_odom_ready(
        tracker, [(1.0, 0.0)], OdomPose(0.0, 0.0, 0.0), 0.1
    )
    assert updated
    assert tracker.tracks[0].id == 1
