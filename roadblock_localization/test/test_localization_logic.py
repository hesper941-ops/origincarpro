import math

import pytest

from roadblock_localization.roadblock_ground_localizer import (
    detection_to_ground_pixel,
    rank_by_distance,
)


def test_bbox_bottom_center():
    assert detection_to_ground_pixel(100, 20, 80, 120) == (140.0, 140.0)
    with pytest.raises(ValueError):
        detection_to_ground_pixel(0, 0, 0, 10)


def test_ids_are_per_frame_distance_rank():
    raw = [
        {"x": 1.2, "y": 0.1, "distance": math.hypot(1.2, 0.1), "confidence": 0.9},
        {"x": 0.65, "y": -0.05, "distance": math.hypot(0.65, -0.05), "confidence": 0.8},
        {"x": 0.85, "y": 0.3, "distance": math.hypot(0.85, 0.3), "confidence": 0.7},
    ]
    ranked = rank_by_distance(raw)
    assert [item["id"] for item in ranked] == [1, 2, 3]
    assert [item["x"] for item in ranked] == [0.65, 0.85, 1.2]
    assert ranked[0]["distance"] < ranked[1]["distance"] < ranked[2]["distance"]
    assert "id" not in raw[0]


def test_empty_frame_stays_empty():
    assert rank_by_distance([]) == []
