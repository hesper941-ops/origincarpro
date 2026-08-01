"""Deterministic quality selection for the three fixed person-board crops."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2


class ImageSelectionError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class ImageQualityScore:
    filename: str
    valid: bool
    rejection_reason: str
    sharpness: float
    sharpness_score: float
    yolo_confidence: float
    confidence_score: float
    crop_area: int
    area_score: float
    final_quality_score: float
    image_path: str
    image_width: int
    image_height: int
    file_size: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SelectionResult:
    selected_image_index: int
    selected_image_path: str
    selected_quality_score: float
    selection_reason: str
    scores: List[ImageQualityScore]

    def selected_score(self) -> ImageQualityScore:
        return self.scores[self.selected_image_index - 1]


def _number(frame: Dict[str, Any], name: str) -> float:
    value = frame.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ImageSelectionError(f"manifest_{name}_invalid")
    value = float(value)
    if not math.isfinite(value):
        raise ImageSelectionError(f"manifest_{name}_invalid")
    return value


def select_best_image(
    frames: Sequence[Dict[str, Any]],
    image_paths: Sequence[Path],
    *,
    sharpness_weight: float,
    confidence_weight: float,
    area_weight: float,
    sharpness_saturation: float,
    minimum_sharpness: float,
    minimum_crop_width: int,
    minimum_crop_height: int,
) -> SelectionResult:
    if len(frames) != 3 or len(image_paths) != 3:
        raise ImageSelectionError("selection_requires_three_images")
    raw: List[Tuple[ImageQualityScore, int]] = []
    for index, (frame, path) in enumerate(zip(frames, image_paths), start=1):
        expected = f"crop_{index:02d}.jpg"
        reason = ""
        sharpness = confidence = 0.0
        crop_width = crop_height = 0
        file_size = image_width = image_height = 0
        try:
            if path.name != expected:
                raise ImageSelectionError("image_filename_invalid")
            for field in ("bbox", "padded_bbox"):
                value = frame.get(field)
                if not isinstance(value, list) or len(value) != 4:
                    raise ImageSelectionError(f"manifest_{field}_invalid")
            sharpness = _number(frame, "sharpness")
            confidence = _number(frame, "yolo_confidence")
            crop_width = int(_number(frame, "crop_width"))
            crop_height = int(_number(frame, "crop_height"))
            if not path.is_file():
                raise ImageSelectionError("image_not_found")
            file_size = path.stat().st_size
            if file_size <= 0:
                raise ImageSelectionError("image_empty")
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ImageSelectionError("image_decode_failed")
            image_height, image_width = image.shape[:2]
            if image_width <= 0 or image_height <= 0:
                raise ImageSelectionError("image_dimensions_invalid")
            if sharpness < minimum_sharpness:
                raise ImageSelectionError("sharpness_below_minimum")
            if not 0.0 <= confidence <= 1.0:
                raise ImageSelectionError("yolo_confidence_invalid")
            if crop_width < minimum_crop_width:
                raise ImageSelectionError("crop_width_below_minimum")
            if crop_height < minimum_crop_height:
                raise ImageSelectionError("crop_height_below_minimum")
        except (OSError, ImageSelectionError) as exc:
            reason = exc.reason if isinstance(exc, ImageSelectionError) else "image_stat_failed"
        area = max(0, crop_width * crop_height)
        raw.append(
            (
                ImageQualityScore(
                    expected, not reason, reason, sharpness, 0.0, confidence,
                    min(1.0, max(0.0, confidence)), area, 0.0, 0.0,
                    str(path), image_width, image_height, file_size,
                ),
                index,
            )
        )
    valid = [item for item, _ in raw if item.valid]
    if not valid:
        raise ImageSelectionError("all_images_invalid")
    max_area = max(item.crop_area for item in valid)
    for item in valid:
        item.sharpness_score = item.sharpness / (
            item.sharpness + sharpness_saturation
        )
        item.area_score = item.crop_area / max_area
        item.final_quality_score = (
            sharpness_weight * item.sharpness_score
            + confidence_weight * item.confidence_score
            + area_weight * item.area_score
        )
    ranked = sorted(
        ((item, index) for item, index in raw if item.valid),
        key=lambda pair: (
            -pair[0].final_quality_score,
            -pair[0].sharpness,
            -pair[0].yolo_confidence,
            -pair[0].crop_area,
            pair[1],
        ),
    )
    selected, selected_index = ranked[0]
    return SelectionResult(
        selected_index,
        selected.image_path,
        round(selected.final_quality_score, 9),
        "highest_quality_score_with_deterministic_tie_break",
        [item for item, _ in raw],
    )
