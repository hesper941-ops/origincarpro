"""Bounding-box helpers for person-board tracking and crop validation."""

import math
from typing import Optional, Sequence, Tuple

import numpy as np


BBox = Tuple[int, int, int, int]


def clip_bbox(box: Sequence[float], width: int, height: int) -> Optional[BBox]:
    if len(box) != 4 or width <= 0 or height <= 0:
        return None
    values = [float(value) for value in box]
    if not all(math.isfinite(value) for value in values):
        return None
    x1 = max(0, min(width, int(math.floor(values[0]))))
    y1 = max(0, min(height, int(math.floor(values[1]))))
    x2 = max(0, min(width, int(math.ceil(values[2]))))
    y2 = max(0, min(height, int(math.ceil(values[3]))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def bbox_size(box: Sequence[float]) -> Tuple[float, float]:
    return max(0.0, float(box[2]) - float(box[0])), max(
        0.0, float(box[3]) - float(box[1])
    )


def bbox_area(box: Sequence[float]) -> float:
    width, height = bbox_size(box)
    return width * height


def short_side(box: Sequence[float]) -> float:
    return min(bbox_size(box))


def normalized_area(box: Sequence[float], width: int, height: int) -> float:
    image_area = float(width * height)
    return bbox_area(box) / image_area if image_area > 0.0 else 0.0


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    intersection = max(
        0.0,
        min(float(first[2]), float(second[2])) - max(float(first[0]), float(second[0])),
    ) * max(
        0.0,
        min(float(first[3]), float(second[3])) - max(float(first[1]), float(second[1])),
    )
    union = bbox_area(first) + bbox_area(second) - intersection
    return intersection / union if union > 0.0 else 0.0


def center_shift_ratio(first: Sequence[float], second: Sequence[float]) -> float:
    first_center = (
        (float(first[0]) + float(first[2])) / 2.0,
        (float(first[1]) + float(first[3])) / 2.0,
    )
    second_center = (
        (float(second[0]) + float(second[2])) / 2.0,
        (float(second[1]) + float(second[3])) / 2.0,
    )
    reference = max(short_side(first), 1.0)
    return (
        math.hypot(
            first_center[0] - second_center[0], first_center[1] - second_center[1]
        )
        / reference
    )


def padded_bbox(
    box: Sequence[float], ratio: float, width: int, height: int
) -> Optional[BBox]:
    box_width, box_height = bbox_size(box)
    expanded = (
        float(box[0]) - box_width * ratio,
        float(box[1]) - box_height * ratio,
        float(box[2]) + box_width * ratio,
        float(box[3]) + box_height * ratio,
    )
    return clip_bbox(expanded, width, height)


def edge_touch_ratio(box: Sequence[float], width: int, height: int) -> float:
    """Return the largest fraction of a bbox side clipped by an image edge.

    A box touching a boundary receives a value based on how much of its own
    corresponding dimension lies outside the image; a merely flush box has a
    small non-zero sentinel so callers can distinguish it from an interior box.
    """
    box_width, box_height = bbox_size(box)
    if box_width <= 0.0 or box_height <= 0.0:
        return 1.0
    overflow_x = max(0.0, -float(box[0])) + max(0.0, float(box[2]) - width)
    overflow_y = max(0.0, -float(box[1])) + max(0.0, float(box[3]) - height)
    ratio = max(overflow_x / box_width, overflow_y / box_height)
    if ratio == 0.0 and (
        float(box[0]) <= 0.0
        or float(box[1]) <= 0.0
        or float(box[2]) >= width - 1
        or float(box[3]) >= height - 1
    ):
        return 1.0 / max(box_width, box_height)
    return ratio


def select_best_detection(
    boxes: np.ndarray,
    scores: np.ndarray,
    image_width: int,
    image_height: int,
    minimum_confidence: float,
    minimum_area: float,
) -> Optional[Tuple[BBox, float]]:
    """Select one target with weighted area, center, confidence, and completeness."""
    candidates = []
    diagonal = max(math.hypot(image_width, image_height) / 2.0, 1.0)
    image_center = (image_width / 2.0, image_height / 2.0)
    for raw_box, raw_score in zip(boxes, scores):
        score = float(raw_score)
        if score < minimum_confidence or bbox_area(raw_box) < minimum_area:
            continue
        clipped = clip_bbox(raw_box, image_width, image_height)
        if clipped is None:
            continue
        raw_area = bbox_area(raw_box)
        completeness = min(1.0, bbox_area(clipped) / max(raw_area, 1.0))
        cx = (clipped[0] + clipped[2]) / 2.0
        cy = (clipped[1] + clipped[3]) / 2.0
        center_score = max(
            0.0, 1.0 - math.hypot(cx - image_center[0], cy - image_center[1]) / diagonal
        )
        area_score = min(
            1.0, normalized_area(clipped, image_width, image_height) / 0.25
        )
        rank = (
            0.40 * area_score + 0.25 * center_score + 0.25 * score + 0.10 * completeness
        )
        candidates.append((rank, clipped, score))
    if not candidates:
        return None
    _, selected_box, selected_score = max(candidates, key=lambda item: item[0])
    return selected_box, selected_score
